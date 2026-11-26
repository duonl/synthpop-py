# Logging

## 1. Introduction
The user needs as a way to see that the synthesis is running. The user also need to know what has happened during synthesis.
This information can be used to troubleshoot or to understand synthpop-py.

## 2. Input and output
The logging should not contain lots of data. 
The user provides no input for logging.

The log messages are passed on to the `logging` module.
### Option 1: library-style
Synthpop-py does nothing further with the log messages.
Filtering, formatting, and displaying is left to the user to configure.
This results in the default behaviour, which is that warnings are printed to the console