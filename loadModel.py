from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="stabilityai/stable-diffusion-2-1",
    local_dir="/media/philip/Games/Users/phili/Documents/.cache/huggingface/hub/modles--stabilityai--stable-diffusion-2-1",
    local_dir_use_symlinks=False
)