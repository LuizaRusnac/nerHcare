# nerHcare

**nerHcare** is a sophisticated Natural Language Processing (NLP) toolkit designed for the extraction and classification of medical entities from unstructured healthcare text data. By leveraging advanced Named Entity Recognition (NER) techniques, nerHcare aims to streamline information extraction for healthcare professionals and researchers.

## Table of Contents
- [Features](#features)
- [Installation](#installation)
- [Model Download](#model-download)
- [Usage](#usage)
- [Documentation](#documentation)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Features

- **Accurate Entity Recognition**: Efficiently identifies and classifies medical entities such as diseases, medications, symptoms, and more.
- **Customizable Models**: Users can tailor models for specific healthcare domains, improving specificity and relevance.
- **User-Friendly Interface**: Designed for ease of use, allowing practitioners and researchers to focus on insights rather than technical details.
- **Comprehensive Data Output**: Generates structured output for easy integration with other data systems and analyses.

## Installation

To install and set up nerHcare, follow these steps:

1. **Clone the repository:**
    ```bash
    git clone https://github.com/LuizaRusnac/nerHcare.git
    ```

2. **Navigate to the project directory:**
    ```bash
    cd nerHcare
    ```

3. **Install the required dependencies:**
    ```bash
    python -m venv env
    .env\Scripts\activate
    pip install -r requirements.txt
    ```

## Model Download

Due to the large size of the model, it has been stored as a zip file in Google Drive. To download the model, follow these steps:

1. Click on the following link to access the model: [Download Model](https://drive.google.com/file/d/1PfPlZS3O1i2UzMjQGgZnA_BBuU-_MiZj/view?usp=drive_link).
2. Once downloaded, unzip the file in the directory within the nerHcare project folder.

Ensure that the model files are correctly referenced in your configuration settings before running the application.

## Usage

To run the nerHcare application using `uvicorn`, execute the following command in your terminal:

```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

Parameters
app: The name of your ASGI application (replace with the appropriate entry point if different).
--host: Specifies the host address (0.0.0.0 makes it accessible from any IP).
--port: The port on which the application will run (default is 8000).
--reload: Enables auto-reload for development for easier testing and debugging.
Ensure that you have your input data correctly configured to interact with the API endpoints of the application.

Acces http://127.0.0.1:8000/ for the interface.

## License

nerHcare is licensed under the MIT License. See the [LICENSE](./LICENSE) file for more information.

