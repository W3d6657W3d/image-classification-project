# Job Application Materials

This document summarizes the project for resumes and interviews. It focuses on completed work only.

## Chinese Resume Version

**垃圾图像分类训练与部署系统 | PyTorch, FastAPI, Streamlit, Docker**

- 基于 12 类垃圾分类数据集构建端到端机器学习工程项目，覆盖数据处理、模型训练、评估、推理服务、前端展示和 Docker 本地部署。
- 对原始不均衡图像数据进行类别统计、基础校验和均衡采样，构建 5,040/1,080/1,080 的 train/validation/test 划分，保证测试集与训练过程隔离。
- 使用 PyTorch 和 torchvision 基于 MobileNetV2 进行迁移学习训练，输出模型 checkpoint、训练历史、测试集指标和混淆矩阵；测试集 Accuracy 0.9537，Macro F1 0.9538。
- 使用 FastAPI 封装模型推理能力，支持图片上传、Top-K 预测、健康检查、类别查询和预测历史查询，并使用 SQLite 保存预测记录。
- 使用 Streamlit 实现轻量前端页面，支持图片预览、Top-K 参数选择、预测概率展示、推理耗时展示和最近预测历史查看。
- 编写 Dockerfile 与 docker-compose.yml，将后端和前端拆分为独立服务，并通过 volume 挂载模型、日志和上传文件，实现可复现的本地演示部署。

## English Resume Version

**Garbage Image Classification Training and Deployment System | PyTorch, FastAPI, Streamlit, Docker**

- Built an end-to-end machine learning engineering project for 12-class garbage image classification, covering data preparation, model training, evaluation, inference serving, demo UI, and Docker-based local deployment.
- Analyzed and processed an imbalanced image dataset, created a balanced train/validation/test split of 5,040/1,080/1,080 images, and kept the held-out test set separate for final evaluation.
- Fine-tuned a MobileNetV2 baseline with PyTorch and torchvision, generated training history, test metrics, and confusion matrix; achieved 0.9537 accuracy and 0.9538 macro F1 on the test split.
- Developed a FastAPI inference service with image upload, Top-K prediction, health check, class lookup, and prediction history APIs, backed by SQLite for runtime records.
- Implemented a Streamlit frontend for image preview, configurable Top-K prediction, probability display, inference latency, backend health status, and recent prediction history.
- Containerized the project with Docker and Docker Compose, running backend and frontend as separate services while mounting model checkpoints and runtime artifacts through volumes.

## One-Minute Chinese Interview Introduction

这个项目是一个 12 类垃圾图像分类的机器学习工程展示项目。我没有只停留在 notebook 训练模型，而是把它做成了一个完整的小型系统：首先对原始不均衡数据集做类别统计、图像校验和均衡采样，得到 train、validation、test 三个划分；然后用 PyTorch 基于 MobileNetV2 做迁移学习训练，并在独立测试集上评估 accuracy、macro precision、macro recall、macro F1 和混淆矩阵，目前测试集 accuracy 是 0.9537，macro F1 是 0.9538。

模型训练完成后，我把 checkpoint 封装成 FastAPI 推理服务，支持图片上传、Top-K 预测、健康检查、类别查询和预测历史；预测记录用 SQLite 保存。前端部分用 Streamlit 做了一个可演示页面，可以上传图片、选择 Top-K、查看预测概率、推理耗时和历史记录。最后我用 Docker Compose 把后端和前端拆成两个服务，模型和日志通过 volume 挂载，方便在本地复现部署。

这个项目主要想体现的是我对机器学习工程闭环的理解，包括数据处理、模型训练、评估、服务化、前端展示、部署和文档化，而不是单纯追求复杂模型。

## Interview Questions and Answer Ideas

### 1. Why did you choose MobileNetV2?

MobileNetV2 is lightweight, stable, and suitable for local CPU or low-resource training and inference. For this portfolio project, the goal was to complete a reliable engineering workflow first, so a lightweight transfer-learning baseline was more appropriate than training a large CNN from scratch.

### 2. How did you handle class imbalance?

The raw dataset had obvious imbalance, especially classes such as `clothes` and `shoes`. I used a balanced subset strategy with up to 600 images per class, then split each class into train/validation/test with a 70/15/15 ratio and random seed 42. This made the first baseline easier to train locally and easier to compare across classes.

### 3. Why use macro F1 instead of only accuracy?

Accuracy gives an overall view, but it can hide class-level problems, especially when datasets are imbalanced. Macro F1 treats each class equally, so it is better for checking whether the model performs consistently across all 12 categories.

### 4. How do you know the reported score is reliable?

The final metrics are computed on a held-out test split that is not used during training or validation model selection. The report also includes per-class precision, recall, F1, and a confusion matrix, so the evaluation is not based on a single aggregate number.

### 5. Which classes were relatively harder?

From the current test report, `plastic` and `white-glass` have lower F1 than stronger classes such as `trash`, `clothes`, and `cardboard`. This suggests there may be visual overlap between some material categories, and future improvements could focus on more data, stronger augmentation, or model comparison.

### 6. What does the FastAPI backend do?

The backend loads the trained checkpoint at startup, exposes health and class endpoints, accepts image uploads through `/predict`, returns Top-K class probabilities, saves uploaded files, and writes prediction metadata to SQLite. It also validates file extensions and returns clear API errors for unsupported inputs.

### 7. Why did you add SQLite?

SQLite is enough for a local demo and keeps the project simple. It demonstrates basic persistence for prediction history without introducing unnecessary infrastructure such as PostgreSQL or a full user system.

### 8. Why use Streamlit for the frontend?

Streamlit is fast to build and suitable for ML demos. The frontend is not meant to show complex UI engineering; it is meant to make the model easy to test by uploading images, changing Top-K, checking probabilities, and viewing recent prediction history.

### 9. What does Docker Compose solve here?

Docker Compose makes the demo easier to reproduce by starting the FastAPI backend and Streamlit frontend as separate services. The model, logs, and uploads are mounted as volumes, so runtime artifacts stay outside the image and the image remains smaller.

### 10. What are the current limitations?

This is not a production high-concurrency system. It does not include cloud deployment, monitoring, user authentication, model version management, or automated retraining. The current goal is a compact, runnable ML engineering baseline that demonstrates the full workflow clearly.

### 11. How would you improve it next?

Reasonable next steps would be model comparison with ResNet18 or EfficientNet-B0, stronger data augmentation, error analysis for confused classes, API tests, basic CI, model version metadata, and optional cloud deployment. I would prioritize based on the target role and time available.

### 12. What engineering ability does this project demonstrate?

It demonstrates that I can take a model beyond training code: prepare data, train and evaluate a baseline, expose inference through an API, build a usable demo UI, persist runtime records, containerize the services, and write documentation that makes the project reproducible.
