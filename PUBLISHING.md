# PyPI 发布指南

本文档描述了如何将 MedVision-Classification 项目发布到 PyPI。

## 前置条件

### 1. 安装发布工具

```bash
pip install -r requirements-publish.txt
```

或者手动安装：

```bash
pip install build twine toml
```

### 2. PyPI 账户设置

1. 在 [PyPI](https://pypi.org/) 注册账户
2. 在 [TestPyPI](https://test.pypi.org/) 注册账户（用于测试）
3. 配置 API tokens（推荐）或设置用户名密码

#### 配置 API Token（推荐）

1. 在 PyPI 账户设置中生成 API token
2. 创建 `~/.pypirc` 文件：

```ini
[distutils]
index-servers = 
    pypi
    testpypi

[pypi]
repository = https://upload.pypi.org/legacy/
username = __token__
password = pypi-your-api-token-here

[testpypi]
repository = https://test.pypi.org/legacy/
username = __token__
password = pypi-your-test-api-token-here
```

## 发布步骤

### 1. 准备发布

1. 更新版本号（在 `pyproject.toml` 中）
2. 更新 `CHANGELOG.md`
3. 确保所有测试通过
4. 确保文档是最新的

### 2. 测试发布

先发布到 TestPyPI 进行测试：

```bash
./release.sh --test
```

测试安装：

```bash
pip install --index-url https://test.pypi.org/simple/ medvision-classification
```

### 3. 正式发布

如果测试发布没问题，发布到正式 PyPI：

```bash
./release.sh
```

### 4. 验证发布

检查包是否正确发布：

```bash
pip install medvision-classification
```

## 脚本选项

`release.sh` 脚本支持以下选项：

- `--test`: 发布到 TestPyPI
- `--dry-run`: 只构建包，不上传
- `--check-only`: 只检查配置，不构建
- `--help`: 显示帮助信息

## 手动发布步骤

如果不想使用脚本，可以手动执行以下步骤：

### 1. 检查配置

```bash
python -m build --sdist --wheel
python -m twine check dist/*
```

### 2. 上传到 TestPyPI

```bash
python -m twine upload --repository testpypi dist/*
```

### 3. 上传到 PyPI

```bash
python -m twine upload dist/*
```

## 常见问题

### 1. 版本冲突

如果提示版本已存在，需要在 `pyproject.toml` 中增加版本号。

### 2. 认证失败

检查 `~/.pypirc` 配置是否正确，或者在命令行输入用户名密码。

### 3. 包结构问题

确保 `medvision_cls` 目录包含 `__init__.py` 文件。

### 4. 依赖问题

确保 `pyproject.toml` 中的依赖版本要求合理。

## 版本管理

建议使用语义化版本（Semantic Versioning）：

- `MAJOR.MINOR.PATCH`
- `MAJOR`: 不兼容的 API 更改
- `MINOR`: 向后兼容的功能添加
- `PATCH`: 向后兼容的错误修复

## 发布检查清单

发布前请确认：

- [ ] 版本号已更新
- [ ] CHANGELOG.md 已更新
- [ ] 所有测试通过
- [ ] 文档是最新的
- [ ] 在 TestPyPI 测试成功
- [ ] README.md 包含正确的安装说明
- [ ] 许可证文件存在
- [ ] 作者信息正确
