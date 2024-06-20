# MinkowskiEngine RunPod Image

* Starting from the exact same RunPod image, where ME can be run successfully
    * `runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04`
* Some packages pre-compiled in teh `wheels` folder
    * `blinker-1.4` needs to be re-installed so that it can be upgraded. URL coded in the Dockerfile.
    * `pytorch3d-0.7.6-cp310-cp310-linux_x86_64.whl` - compiled on RunPod
    * `MinkowskiEngine-0.5.4-cp310-cp310-linux_x86_64.whl` - compiled locally using another Docker
