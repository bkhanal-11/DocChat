# DocChat

DocChat is a Streamlit-based application that enables users to interact with their documents in a conversational manner. By leveraging the power of OpenAI's GPT models, DocChat allows users to upload documents and engage in a chat-like interface to extract information, ask questions, or simply explore the content of their documents in a more intuitive way.

## Features

- **OpenAI Integration**: Utilize various OpenAI models including GPT-3.5 and GPT-4 for understanding and generating responses based on the document's content.
- **Support for Multiple Document Types**: Upload and interact with PDFs, DOCXs, Markdown, and text files.
- **Session Management**: Unique session handling for each user, ensuring a personalized and secure experience.
- **Easy to Use Interface**: A simple and intuitive UI/UX, making it accessible for users with varying levels of technical expertise.

## Getting Started

### Prerequisites

Before you begin, ensure you have met the following requirements:

- Python 3.7 or later
- An OpenAI API key. You can obtain one by signing up at [OpenAI](https://platform.openai.com/account/api-keys).

### Installation

1. Clone the repository:

```bash
git clone https://github.com/bkhanal-11/DocChat.git
cd DocChat
python3 -m venv venv
source venv/bin/activate
pip3 install -r requirements.txt
streamlit run app.py
```

After running the command, Streamlit will start the application and provide a local URL which you can visit using your web browser to access DocChat.

### Configuration

- **OpenAI API Key**: Enter your OpenAI API key in the sidebar within the application to enable the document processing features.
- **Document Upload**: Supported document types include PDF, DOCX, Markdown, and text files. You can upload multiple documents at once.

## Contributing

We welcome contributions to DocChat! If you have suggestions or improvements, feel free to fork the repository and submit a pull request. You can also open an issue if you encounter any problems or have any questions.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

