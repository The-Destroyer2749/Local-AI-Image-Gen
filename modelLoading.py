from huggingface_hub import snapshot_download, try_to_load_from_cache, _CACHED_NO_EXIST, list_repo_files
import os

debug = False


cachedModelNames = {
    "FLUX.1 Schnell": "models--black-forest-labs--FLUX.1-schnell",
    "StableDiffusion_3.5_medium": "models--stabilityai--stable-diffusion-3.5-medium",
    "StableDiffusion_2.1": "models--stabilityai--stable-diffusion-2-1",
    "StableDiffusion_1.4": "models--CompVis--stable-diffusion-v1-4"
}

modelNames = {
    "FLUX.1 Schnell": "black-forest-labs/FLUX.1-schnell",
    "StableDiffusion_3.5_medium": "stabilityai/stable-diffusion-3.5-medium",
    "StableDiffusion_2.1": "stabilityai/stable-diffusion-2-1",
    "StableDiffusion_1.4": "CompVis/stable-diffusion-v1-4"
}

def checkIfModelIsCached(repoId, cacheDir=None):
    modelDir = os.path.join(cacheDir, cachedModelNames[repoId])

    # localRepoFiles = os.listdir(modelDir)
    # localRepoSaftensorFiles = [file for file in localRepoFiles if file.endswith('.safetensors')]

    # TODO: replace this with proper file searching instead of just grabbing the first one
    # filename = localRepoSaftensorFiles[0]

    # print(filename)

    if cacheDir is not None:
        filepath = try_to_load_from_cache(repo_id=modelNames[repoId], cache_dir=cacheDir, filename="config.json") # uses custom directory for cached models
    else:
        filepath = try_to_load_from_cache(repo_id=modelNames[repoId], filename="config.json") # uses default directory for cached models

    if isinstance(filepath, str):
        if debug:
            print("File cached proceeding...")

        if cacheDir is not None:
            return modelDir
        else:
            return modelNames[repoId]
    elif filepath is _CACHED_NO_EXIST:
        raise RuntimeError("ERROR: _CACHED_NO_EXIST (not really sure what that means though)")
    else:
        print("Downloading model...")

        if cacheDir is not None:
            snapshot_download(
                repo_id=repoId,
                local_dir=modelDir,
                local_dir_use_symlinks=False
            )
            return modelDir
        else:
            snapshot_download(
                repo_id=repoId,
                local_dir_use_symlinks=False
            )
            return modelNames[repoId]