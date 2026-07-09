import os, torch
from tqdm import tqdm
from accelerate import Accelerator
from .training_module import DiffusionTrainingModule
from src.training_module import MetaViewTrainingModule
from .logger import ModelLogger

from PIL import Image


def collate_fn(batch):
    if len(batch) == 0:
        return {}

    collated = {}
    keys = batch[0].keys()
    for key in keys:
        values = [sample[key] for sample in batch]
        first_val = values[0]
        if isinstance(first_val, torch.Tensor):
            # 对于 Tensor，使用 torch.stack 沿第0维堆叠（要求所有张量形状相同）
            collated[key] = torch.stack(values, dim=0)
        elif isinstance(first_val, Image.Image):
            collated[key] = values
        elif isinstance(first_val, str):
            collated[key] = values
        else:
            collated[key] = values

    return collated

def launch_training_task(
    accelerator: Accelerator,
    dataset: torch.utils.data.Dataset,
    model: MetaViewTrainingModule,
    model_logger: ModelLogger,
    learning_rate: float = 1e-5,
    weight_decay: float = 1e-2,
    num_workers: int = 1,
    save_steps: int = None,
    num_epochs: int = 1,
    batch_size: int = 1,
    args = None,
):
    if args is not None:
        learning_rate = args.learning_rate
        weight_decay = args.weight_decay
        num_workers = args.dataset_num_workers
        save_steps = args.save_steps
        num_epochs = args.num_epochs
    
    optimizer = torch.optim.AdamW(model.trainable_modules(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)
    # dataloader = torch.utils.data.DataLoader(dataset, shuffle=False, collate_fn=lambda x: x[0], num_workers=num_workers)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=num_workers)

    model, optimizer, dataloader, scheduler = accelerator.prepare(model, optimizer, dataloader, scheduler)
    
    def grad_hook(name):
        def hook(grad):
            # # print(f"{name} gradient norm:{grad.norm().item():.6f}")
            # if grad.norm().item() > 1:
            #     print(f" gradient over 1: {name} {grad.norm().item():.6f}")
            if torch.isnan(grad).any():
                print(f"!!! NaN gradient: {name}")
            if torch.isinf(grad).any():
                print(f"!!! Inf gradient: {name}")
            return grad
        return hook

    for name, param in model.named_parameters():
        if param.requires_grad:
            param.register_hook(grad_hook(name))

    NaN_step = 0
    for epoch_id in range(num_epochs):
        for data in tqdm(dataloader):
            with accelerator.accumulate(model):
                # print(type(data))
                # print(data["prompt"])
                optimizer.zero_grad()
                if dataset.load_from_cache:
                    loss = model({}, inputs=data)
                else:
                    loss = model(data)

                if torch.isnan(loss).any():
                    print(f"!!! Loss is NaN at step {model_logger.num_steps}! Skipping...")
                    NaN_step += 1
                    print(data["name"])
                    exit(0)
                    
                accelerator.backward(loss)
                
                max_norm = 5.0
                if accelerator.sync_gradients:
                    grad_norm = accelerator.clip_grad_norm_(model.parameters(), max_norm=max_norm)
                    if accelerator.is_main_process:
                        if grad_norm > 5.0:
                            print(f"gradient over 5: {grad_norm:.4f}")

                optimizer.step()
                model_logger.on_step_end(accelerator, model, save_steps)
                scheduler.step()
        if save_steps is None:
            model_logger.on_epoch_end(accelerator, model, epoch_id)
    model_logger.on_training_end(accelerator, model, save_steps)


def launch_data_process_task(
    accelerator: Accelerator,
    dataset: torch.utils.data.Dataset,
    model: DiffusionTrainingModule,
    model_logger: ModelLogger,
    num_workers: int = 8,
    args = None,
):
    if args is not None:
        num_workers = args.dataset_num_workers
        
    dataloader = torch.utils.data.DataLoader(dataset, shuffle=False, collate_fn=lambda x: x[0], num_workers=num_workers)
    model, dataloader = accelerator.prepare(model, dataloader)
    
    for data_id, data in enumerate(tqdm(dataloader)):
        with accelerator.accumulate(model):
            with torch.no_grad():
                folder = os.path.join(model_logger.output_path, str(accelerator.process_index))
                os.makedirs(folder, exist_ok=True)
                save_path = os.path.join(model_logger.output_path, str(accelerator.process_index), f"{data_id}.pth")
                data = model(data)
                torch.save(data, save_path)
