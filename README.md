# AccAltProj for robust PCA

<p align="center">
  <a href="https://github.com/loiccoyle/RPCA/actions/workflows/ci.yml"><img src="https://github.com/loiccoyle/RPCA/actions/workflows/ci.yml/badge.svg"></a>
  <a href="https://pypi.org/project/rpca/"><img src="https://img.shields.io/pypi/v/rpca"></a>
  <a href="./LICENSE.md"><img src="https://img.shields.io/badge/license-MIT-blue.svg"></a>
  <img src="https://img.shields.io/badge/platform-linux%20%7C%20macOS%20%7C%20windows-informational">
</p>

> Port of the AccAltProj algorithm for robust PCA to python.

<div align="center">
  <image src="https://github.com/loiccoyle/RPCA/assets/33181239/dbccf187-740f-461f-8e05-78ad497b2d30" />
</div>

This is a python port of the [AccAltProj algorithm for robust PCA](https://github.com/caesarcai/AccAltProj_for_RPCA), described in this [paper](https://arxiv.org/abs/1711.05519).

This implementation follows `sklearn`'s `fit` & `transform` API.

## 📦 Installation

Requires python 3

In a terminal:

```sh
pip install rpca
```

As always, it is usually a good idea to use a [virtual environment](https://docs.python.org/3/library/venv.html).
