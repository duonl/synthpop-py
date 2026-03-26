# About this documentation

We document for both users and developers. 
The [api docs](./api%20reference/api%20docs%20index.rst) is for users quickly wanting to read the documentation of a specific method.
Users new to this package should look at the [examples](./user%20guides/examples.md).
The [developer documentation](./developer/developer_index.md) is for developers that want to understand this package or know more about how we develop.
The [functional descriptions](./functional%20descriptions/fd_index.md) aim to describe what this package is designed to do, independent of programming language. These documents act as the blue print when developing features. The standards and norms described in the developer documentation and the functional descriptions should be enough to implement a feature. 

## The documentation required for new features
The first documentation that should exist is a functional description. Ideally, this should be written before any code, as that ensures that the code is tailored to the requirements and not the other way around
Docstrings are required for all public classes, methods and functions. For internal-only components, docstrings are optional.
Extra user guides and examples are not always needed, but should be provided if the user has to do "new" things to use this feature. 

## About writing documentation.
We aim to write documentation in a way that is easy to produce and can be automaticalyl converted into a more readable format
For now, we use markdown to write most documentation. Only the docstrings in the code are in [reStructuredText](https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html). The flavor of markdown that we use is [MyST](https://myst-parser.readthedocs.io/en/latest/syntax/typography.html), which stands for Markedly StructuredText. MyST is a strict superset of [CommonMark](https://commonmark.org/).

MyST has been configured with `dollarmath` and `amsmath` enabled. The `dollarmath` extension allows inline LaTeX expressions using dollar-sign syntax, like this: `$a^2 = b^2 + c^2$` to write $a^2 = b^2 + c^2$. The `amsmath` extension enables full LaTeX math environments to be used directly in MyST markdown, for example:
```latex
 \begin{gather*}
  a_1=b_1+c_1\\a_2=b_2+c_2-d_2+e_2
  \end{gather*}
```
This renders as 
\begin{gather*}
  a_1=b_1+c_1\\a_2=b_2+c_2-d_2+e_2.
\end{gather*}

Another enabled extension is [Mermaid](https://mermaid.ai/open-source/intro/), which allows you to write diagrams like this:
````
```{mermaid}
graph TD
    A[Start]-->B[End]
```
````
```{mermaid}
graph TD
    A[Start]-->B[End]
```
## About building and publishing documentation

The languages to write documentation are MyST and reStructuredText, but the language to view the documentation online is HTML (and others). The main tool to do the translation is [sphinx](https://www.sphinx-doc.org/en/master/). We use sphinx with the following extensions:
- [autodoc](https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html#ext-autodoc) to include the docstrings in the online documentation.
- [myst_parser](https://myst-parser.readthedocs.io/en/latest/index.html) to enable markdown.
- [mermaid](https://github.com/mgaitan/sphinxcontrib-mermaid) to render diagrams.

The documentation is hosted on [ReadTheDocs.io](https://about.readthedocs.com/), which is a platform specifically for hosting documentation of code. 
Readthedocs.io is configured (via a github app) to build the documentation of the develop branch and of pull requests. It automatically builds the documentation when there is a new push to develop or a (commit to) a pull request. To build the documentation, readthedocs.io first clones the repository. The file `.readthedocs.yaml` specifies how readthedocs.io should build the documentation. 

The first step to building the documentation is installing this package including the `docs` dependency group (`poetry install --with docs`). The package needs to be installed so that autodoc can import every module to access the docstrings. Sphinx then parses everything to HTML. The HTML can then be served on [synthpop-py.readthedocs.io](https://synthpop-py.readthedocs.io)