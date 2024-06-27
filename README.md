# Fake News Detection App

This project is a web application built using Streamlit that helps identify whether a piece of news is fake or real using machine learning models. The app allows users to select a model, input news text, and receive a prediction based on the selected model.

## Project Structure

- `app.py`: The main application file.
- `vectorizer.pkl`: The pickled TF-IDF vectorizer used for text transformation.
- `models/`: Directory containing the saved machine learning models.
- `requirements.txt`: Lists the Python packages required for the project.

## Prerequisites

- Python 3.7 or higher
- Streamlit
- scikit-learn
- joblib
- pyyaml

## Installation

1. **Clone the repository**:
    ```bash
    git clone https://github.com/your-username/fake-news-detection.git
    cd fake-news-detection
    ```

2. **Create a virtual environment**:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3. **Install the required packages**:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. **Run the Streamlit app**:
    ```bash
    streamlit run streamlit_app.py
    ```

2. **Open your web browser** and go to `http://localhost:8501`.

3. **Use the app**:
    - Select a machine learning model from the sidebar.
    - Enter the news text in the provided text area.
    - Click the "Predict" button to get the prediction.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
