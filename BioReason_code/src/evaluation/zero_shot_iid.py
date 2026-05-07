"""
Do zero-shot classification on IID data with both seen and unseen classes.

Single-process. If you want to run all evaluations of a single model at once, look
in scripts/.

Writes the output to a plaintext and JSON format in the logs directory.
"""

import logging
import sys
import torch
import torch.nn.functional as F
from tqdm import tqdm
import os
import json 

from ..open_clip import (
    create_model_and_transforms,
    get_cast_dtype,
    get_tokenizer,
    trace_model,
)
from ..training.imagenet_zeroshot_data import openai_imagenet_template
from ..training.precision import get_autocast

from .data import DatasetFromFile
from .params import parse_args
from .utils import (
    configure_logging,
    configure_torch_backends,
    get_dataloader,
    init_device,
    log_params,
    normalize_force_image_size,
    random_seed,
)


def zero_shot_classifier(model, classnames, templates, args):
    tokenizer = get_tokenizer(args.model)
    with torch.no_grad():
        zeroshot_weights = []
        for classname in tqdm(classnames):
            texts = [template(classname) for template in templates]  # format with class
            texts = tokenizer(texts).to(args.device)  # tokenize
            class_embeddings = model.encode_text(texts)
            class_embedding = F.normalize(class_embeddings, dim=-1).mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).to(args.device)
    return zeroshot_weights


def accuracy(output, target, topk=(1,)):
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return dict([
        (k,float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()))
        for k in topk
    ])


def run(model, classifier, dataloader, args):
    autocast = get_autocast(args.precision)
    cast_dtype = get_cast_dtype(args.precision)
    
    all_predictions = []
    class_names = dataloader.dataset.classes
    
    has_samples = hasattr(dataloader.dataset, 'samples')
    samples = dataloader.dataset.samples if has_samples else []
    global_idx = 0 
    
    # 获取当前具体的 CSV 文件名，用于区分数据块内部的来源
    dataset_base_name = os.path.basename(args.label_filename) 
    
    with torch.no_grad():
        n = 0.0
        topk = dict()
        for i in (1,min(len(dataloader.dataset.classes),3), min(len(dataloader.dataset.classes),5)):
            topk[i] = 0.0
        for images, target in tqdm(dataloader, unit_scale=args.batch_size):
            batch_size = images.size(0)
            images = images.to(args.device) 
            if cast_dtype is not None:
                images = images.to(dtype=cast_dtype)
            target = target.to(args.device)

            with autocast():
                image_features, _ = model.encode_image(images)
                image_features = F.normalize(image_features, dim=-1)
                logits = model.logit_scale.exp() * image_features @ classifier

            k_preds = min(5, logits.shape[1])
            _, top5_indices = logits.topk(k_preds, dim=1)
            
            top5_indices = top5_indices.cpu().tolist()
            target_list = target.cpu().tolist()
            
            for idx in range(batch_size):
                gt_idx = target_list[idx]
                pred_indices = top5_indices[idx]
                
                if has_samples and (global_idx + idx) < len(samples):
                    sample_item = samples[global_idx + idx]
                    # 获取原始路径（可能是相对路径或仅文件名）
                    raw_path = sample_item[0] if isinstance(sample_item, (tuple, list)) else str(sample_item)
                    
                    # 转换为绝对路径
                    # 如果 raw_path 已经是绝对路径，os.path.join 会自动处理
                    full_path = os.path.join(args.data_root, raw_path)
                    img_path = os.path.abspath(full_path)
                else:
                    img_path = "unknown_path"
                
                all_predictions.append({
                    "csv_source": dataset_base_name,
                    "image_path": img_path, # 现在这里是绝对路径了
                    "ground_truth": class_names[gt_idx],
                    "top_5_predictions": [class_names[p_idx] for p_idx in pred_indices]
                })
            # ---------------------------
            
            global_idx += batch_size

            acc = accuracy(logits, target, topk=topk.keys())
            for k,v in acc.items():
                topk[k] += v
            n += images.size(0)

    for k in topk.keys():
        topk[k] /= n
        
    return topk, all_predictions


def zero_shot_eval(model, data, args):
    results = {}
    predictions_log = [] # 改用列表来收集，方便直接追加到文件

    logging.info("Starting zero-shot.")

    for split in data:
        logging.info("Building zero-shot %s classifier.", split)
        classnames = [c for c in data[split].dataset.classes]

        classifier = zero_shot_classifier(
            model, classnames, openai_imagenet_template, args
        )

        topk, split_predictions = run(model, classifier, data[split], args)
        
        # 将本次 split 的所有预测展开合并进主列表
        predictions_log.extend(split_predictions)

        for k,v in topk.items():
            results[f"{split}-top{k}"] = v

        logging.info("Finished zero-shot %s with total %d classes.", split, len(data[split].dataset.classes))

    logging.info("Finished zero-shot.")

    return results, predictions_log


if __name__ == "__main__":
    # === 拦截自定义的 output_group 参数 ===
    output_group = "Misc_Results" # 默认值
    cleaned_argv = []
    i = 1
    while i < len(sys.argv):
        if sys.argv[i] == '--output_group':
            if i + 1 < len(sys.argv):
                output_group = sys.argv[i+1]
                i += 2
            else:
                i += 1
        else:
            cleaned_argv.append(sys.argv[i])
            i += 1
            
    # 将清理过（去除了 output_group）的参数传给原有的解析器，防止报错
    args = parse_args(cleaned_argv)
    # ========================================

    configure_torch_backends(deterministic=False)

    device = init_device(args)

    log_base_path = configure_logging(
        args, "zero_shot_iid", include_workers=True, log_filename="out.log"
    )

    normalize_force_image_size(args)

    random_seed(args.seed, 0)
    model, preprocess_train, preprocess_val = create_model_and_transforms(
        args.model,
        args.pretrained,
        precision=args.precision,
        device=device,
        jit=args.torchscript,
        force_quick_gelu=args.force_quick_gelu,
        force_custom_text=args.force_custom_text,
        force_patch_dropout=None,
        force_image_size=args.force_image_size,
        pretrained_image=args.pretrained_image,
        image_mean=args.image_mean,
        image_std=args.image_std,
        aug_cfg=args.aug_cfg,
        output_dict=True,
    )

    random_seed(args.seed, args.rank)

    if args.trace:
        model = trace_model(model, batch_size=args.batch_size, device=device)

    data = {
        "val-unseen": get_dataloader(
            DatasetFromFile(args.data_root, args.label_filename, transform=preprocess_val, classes=args.text_type),
            batch_size=args.batch_size,num_workers=args.workers
        ),
    }

    model.eval()
    
    metrics, current_predictions = zero_shot_eval(model, data, args)
    
    # === 修改后的保存逻辑：文件夹名 + 文件名，确保唯一性 ===
    current_dir = os.path.dirname(__file__)
    
    # 1. 获取 CSV 所在的文件夹名（例如：PLK_Mini）
    parent_folder_name = os.path.basename(os.path.dirname(args.label_filename))
    
    # 2. 获取 CSV 的纯文件名（例如：labels）
    csv_base_name = os.path.basename(args.label_filename).replace('.csv', '').replace('.metadata', '')
    
    # 3. 组合成唯一的文件名（例如：PLK_Mini_labels.json）
    # 如果文件夹名和文件名一样（比如 rare-species/rare-species.csv），则只保留一个，避免冗余
    if parent_folder_name == csv_base_name:
        dataset_unique_name = csv_base_name
    else:
        dataset_unique_name = f"{parent_folder_name}_{csv_base_name}"
    
    # 4. 设定保存目录：A_out_json/{output_group}/
    save_dir = os.path.normpath(os.path.join(current_dir, f"../../A_out_json/{output_group}"))
    os.makedirs(save_dir, exist_ok=True)
    
    # 5. 最终文件路径
    json_save_path = os.path.join(save_dir, f"{dataset_unique_name}.json")
    
    # 6. 写入结果
    with open(json_save_path, 'w', encoding='utf-8') as f:
        json.dump(current_predictions, f, ensure_ascii=False, indent=4)
        
    logging.info(f"Top-5 预测结果已保存至: {json_save_path}")
    # ==========================================================
    
    dataset_name = os.path.abspath(args.label_filename)
    
    logging.info("\n"*6+"="*60)
    logging.info(f"评估总结 | Evaluation Summary")
    logging.info(f"  数据集 (Dataset): {dataset_name}")
    logging.info(f"  归属板块 (Group): {output_group}")
    logging.info(f"  方法 (Method):   Zero-shot (IID)")
    
    for key, value in metrics.items():
        display_name = key.split('-')[-1].capitalize()
        logging.info(f"  {display_name:10}: {value*100:.2f}%")
    
    logging.info("="*60 + "\n"*6)