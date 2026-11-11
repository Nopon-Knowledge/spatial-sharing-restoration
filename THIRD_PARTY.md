# Third-party dependencies

The MIT license in this repository covers its original code and documentation.
External dependencies retain their own copyright notices and license terms.

## AirNet

- Upstream: https://github.com/XLearning-SCU/2022-CVPR-AirNet
- Authors: Boyun Li, Xiao Liu, Peng Hu, Zhongqin Wu, Jiancheng Lv, Xi Peng
- Pinned revision: `bedea82f7f7a71f135bcc026fd02ea9f2feaa710`
- Reference: *All-in-One Image Restoration for Unknown Corruption*, CVPR 2022.
- Integration: upstream CBDE and DGRN, including MMCV's modulated deformable convolution.

No repository-level license was identified in the inspected upstream revision.
AirNet source is therefore **not redistributed in this repository** or included
under this project's MIT license. `scripts/fetch_airnet.py` obtains the pinned
revision directly from the authors into the ignored `third_party/AirNet`
directory. Original upstream notices remain intact. The source record in
[`docs/AIRNET_SOURCE.json`](docs/AIRNET_SOURCE.json) records the inspected files
and their hashes; it contains no upstream source code.

## Runtime libraries

PyTorch, NumPy, SciPy, Pillow, PyYAML, scikit-image, MMCV, torchvision, and LPIPS
are installed separately. Their code and pretrained weights are not bundled.
See each upstream project for its applicable terms and acknowledgements.

## Data and model weights

WeatherBench images, original split manifests, and pretrained restoration
weights are not distributed here. The smoke command creates synthetic test
fixtures locally; these are not benchmark data or research measurements.
