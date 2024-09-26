# AskLibby - Your Friendly Library Information Assistant

AskLibby is a simple library information chatbot designed to assist users by answering general queries related to the library's services, resources, and general information. AskLibby classifies user queries into various intents and provides accurate, prompt responses based on the categorized intent. The bot is designed to reduce the workload on human staff while enhancing user experience by providing instant answers to common questions.

## Implementation details:

* The implementation is designed to be beginner-friendly, offering a clear introduction to chatbots. 
* It employs a simple Feedforward Neural Network with 2 hidden layers. 
* Adapting the chatbot for your own needs is simple—update the ```intents.json``` file with new patterns and responses, and retrain the model (see further details below).

## Directory Structure <a name="directory-structure"></a>

```
├── intents.json/                              <- AskLibby training data
├── utils.py/                                  <- Utility functions of the project
├── train_AskLibby.py/                         <- Trains the AskLibby chatbot model
├── asklibby_core.keras/                       <- Saved model for the AskLibby chatbot
├── training_data/                             <- Holds the training data for the chatbot
├── AskLibby.py/                               <- Main script for running the AskLibby chatbot
├── LICENSE                                    <- Project's License
├── README.md                                  <- The top-level README for developers using this project
```

## Usage:

Run
```
python train_AskLibby.py
```

This will dump ```asklibby_core.keras``` and ```training_data``` files in the current directory. And then run
 
```
python AskLibby.py
```

## Customize:

Take a look at the intents.json file. You can adjust it to fit your specific use case by adding new tags, patterns, and responses for the chatbot. 
Remember to re-train the model whenever you make changes to this file.

```json
{
  "intents": [
    {
      "tag": "greeting",
      "patterns": [
        "Hi",
        "Hello",
        "How are you doing?",
        "Yo!",
        "Hey there!",
        "Whats up?",
        "Howdy ?",
        "Hey",
        "Good morning",
        "Good afternoon",
        "Good evening",
        "Hi there",
        "Is anyone here?",
        "Anyone available?",
        "Hello, anyone?"
      ],
      "responses": [
        "Hello! How can I assist you with library information today?",
        "Hi there! What can I help you with?",
        "Hey! Need to know something about the library?"
      ]
    },
    ...
  ]
}
```

## License <a name="license"></a>
This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](https://github.com/YuganshG/AskLibby/blob/main/LICENSE) file for details.

## Support & Contact <a name="support-contact"></a>
If you encounter any issues, have questions or just want to chat about machine learning? Feel free to [email](yugansh.goyal101@gmail.com) me or connect with me on [LinkedIn](https://www.linkedin.com/in/yuganshgoyal/).
