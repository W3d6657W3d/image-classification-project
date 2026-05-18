# Task Prompts

Use these prompts when opening focused conversations for later stages. Keep the
current conversation as the project initialization and coordination thread.

## Common Context

```text
This is a machine learning engineering portfolio project.

Project directory:
D:\PycharmProjects\image-classification-project

Project goal:
Finish a lightweight image classification training and deployment system within
two weeks. The project should demonstrate a complete machine learning workflow:
data processing, model training, model evaluation, FastAPI inference service,
frontend display, SQLite prediction history, Docker deployment, and documentation.

Target roles:
Junior machine learning engineer, algorithm application engineer, data science
roles, AI solution engineer, and algorithm application roles in fintech or
banking technology.

Constraints:
The local machine has limited hardware, so prefer lightweight models, transfer
learning, small public datasets, and fast deployment choices.

Positioning:
This is not only an image recognition demo. It should show the ability to take a
machine learning problem from data to model to deployment.
```

## Dataset Conversation

```text
This conversation is only for dataset selection and data analysis.

Please first read the current project structure and docs/project_plan.md. Then
help me choose an image classification dataset suitable for a two-week project,
low-end hardware, and portfolio presentation. After the dataset is selected,
guide me step by step through data folder planning, class distribution analysis,
sample visualization, and train/validation/test split.

Do not give too many manual web or software steps at once. If I need to download
data in a browser, give only the next one or two key steps and wait for feedback.
```

## Training Conversation

```text
This conversation is only for model training and evaluation.

Based on the existing project structure and selected dataset, gradually implement
a transfer learning training pipeline. Prioritize low-end hardware and a stable
baseline first. The output should include accuracy, precision, recall, F1,
confusion matrix, and the best saved model file.
```

## Backend Conversation

```text
This conversation is only for the FastAPI backend inference service.

Based on the existing project structure and trained model file, implement image
upload, model inference, Top-K prediction, error handling, logging, and SQLite
prediction history. Please read the project structure first and then implement
the backend step by step.
```

## Frontend Conversation

```text
This conversation is only for the frontend display page.

Based on the existing FastAPI inference API, implement a simple usable frontend
page with image upload, image preview, Top-K result display, and prediction
history view. Prioritize a deliverable two-week project.
```

## Deployment Conversation

```text
This conversation is only for Docker and deployment.

Based on the current project, complete Dockerfile or docker-compose setup and
help choose a low-cost deployment option. The goal is to make the project easy
for an interviewer to reproduce or try online.
```

## Resume Conversation

```text
This conversation is only for README, project report, and resume packaging.

Based on the finished code, model results, and deployment status, polish the
README, write the model evaluation report, and generate resume bullets for
junior machine learning, algorithm application, data science, and AI solution
engineer roles.
```
