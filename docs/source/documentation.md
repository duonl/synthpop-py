# About this documentation

We document for both users and developers. 
The [api docs](./api%20reference/api%20docs%20index.rst) is for users quickly wanting to read the documentation of a specific method.
Users new to this package should look at the [examples](./user%20guides/examples.md)
The [developer documentation](./developer/developer_index.md) is for developers that want to understand this package or know more about how we develop.
The [functional descriptions](./functional%20descriptions/fd_index.md) aim to describe what this package is designed to do, independent of programming language. These documents act as the blue print when developing features. The standards and norms described in the developer docs and the functional descriptions should be enough to implement a feature. 

## The documentation required for new features
The first documentation that should exists is a functional description. This should ideally be written before any code is written, since that ensures that the code is tailored to the requirements and not the other way around. 
Any class, method, or function that you expect a user to use should have docstrings. Docstrings are optional for internal functions and classes.
Extra user guides and examples are not always needed, but should be provided if the user has to do "new" things to use this feature. 
