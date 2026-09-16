# Contributing to synthpop-py

First off, thank you for your interest in contributing to synthpop-py! ❤️

Contributions of all kinds are welcome, including bug reports, documentation improvements, feature suggestions, code contributions, and improvements to the development process. This guide explains how to contribute and what information we need to review contributions effectively.

Before contributing, please read the section relevant to the type of contribution you would like to make. The community looks forward to your contributions.

## I have a question

> Before opening a question, please consult the available [documentation](https://synthpop-py.readthedocs.io/en/develop/).

You should also search the existing [GitHub issues](https://github.com/duonl/synthpop-py/issues), as your question may already hae been answered or discussed. If you find a relevant issue but require further clarification, you are welcome to add your question to that discussion.

If you still need help, please open a [new GitHub issue](https://github.com/duonl/synthpop-py/issues/new) and provide as much relevant context as possible. Depending on the issue, this may include:
- a description of what you are trying to achieve;
- the code you are using;
- the behaviour you are observing;
- the versions of synthpop-py and relevant dependencies;
- your operating system and Python version.

This information helps us understand and respond to your question as soon as possible.

<!--
You might want to create a separate issue tag for questions and include it in this description. People should then tag their issues accordingly.

Depending on how large the project is, you may want to outsource the questioning, e.g. to Stack Overflow or Gitter. You may add additional contact and information possibilities:
- IRC
- Slack
- Gitter
- Stack Overflow tag
- Blog
- FAQ
- Roadmap
- E-Mail List
- Forum
-->

## I want to contribute

> ### Legal notice <!-- omit in toc -->
> By contributing to this project, you confirm that
> - you have authored the contribution or have the necessary rights to submit it;
> - you have the right to provide the contributed content under the project's [license](LICENSE.md); and
> - your contribution may be distributed under the project's license.

### AI-generated code
AI-assisted contributions are not prohibited. However, contributors remain responsible for everything they submit.

If you use an AI tool to generate or modify code, you must understand the resulting code sufficiently to explain and maintain it during review. You are also responsible for ensuring that your contribution complies with the legal notice above and does not introduce content that cannot be contributed under the project's licence.

**Disclaimer:** This policy may be updated as the use of AI-assisted development evolves.

### Reporting bugs

<!-- omit in toc -->
#### Before submitting a bug report

A useful bug report allows others to understand and reproduce the problem without requiring extensive follow-up questions. Before submitting a report, please investigate the issue and collect as much relevant information as possible.

In particular:
- Make sure you are using a recent version of synthpop-py.
- Check the [documentation](https://synthpop-py.readthedocs.io/en/develop/) to ensure that the behaviour is not caused by an incorrect configuration or unsupported usage.
- Search the existing [GitHub issues](https://github.com/duonl/synthpop-py/issues) to see whether the problem has already been reported or resolved.
- Where appropriate, search for discussions outside the project (like Stack Overflow) that may help identify the cause of the problem.
- Try to create a minimal reproducible example.

Please also collect relevant information, such as:
- the full traceback or error message;
- your operating system and platform;
- your Python version;
- the version of synthpop-py and relevant dependencies;
- the relevant input and output;
- clear steps for reproducing the issue.

If possible, check whether the issue can also be reproduced with earlier versions of the package.

<!-- omit in toc -->
#### Reporting security issues
> **Do not report security vulnerabilities or issues containing sensitive information through the public issue tracker.** Please report security-related issues privately by email to synthetische.data@duo.nl.

<!-- omit in toc -->
#### Submitting a bug report
We use [GitHub issues](https://github.com/duonl/synthpop-py/issues) to track bugs and unexpected behaviour.

When reporting an issue:
- Use a clear and descriptive title.
- Describe the behaviour you expected.
- Describe the behaviour you observed instead.
- Provide clear, step-by-step instructions for reproducing the issue.
- Include a minimal reproducible example where possible.
- Include the relevant environment and version information.
- Include any relevant information collected [before submitting a bug report](#before-submitting-a-bug-report).

Once an [issue](https://github.com/duonl/synthpop-py/issues/new) has been submitted, a member of the project team may ask for additional information or clarification.

If the issue can be reproduced and is confirmed to be a bug, it will be labelled and prioritised accordingly.

<!-- You might want to create an issue template for bugs and errors that can be used as a guide and that defines the structure of the information to be included. If you do so, reference it here in the description. -->

### Suggesting enhancements

We welcome suggestions for new functionality and improvements to existing features, documentation, usability, performance, and code quality.

<!-- omit in toc-->
#### What improvements can I suggest?
Examples of improvements we are happy to include are:
- improvements to documentation, readability, or language;
- performance improvements that maintain backwards compatibility;
- proposals for changes that may introduce breaking changes;
- missing documentation, guides or examples;
- improvements to existing functionality;
- entirely new features.

If a guide, example or feature would have made synthpop-py easier or more useful for you, it may also be valuable to other users.

<!-- omit in toc -->
#### Before submitting an enhancement
Before opening an enhancement request:
- Make sure that you are using a recent version of synthpop-py.
- Read the [documentation](https://synthpop-py.readthedocs.io/en/develop/) to determine whether the functionality already exists or can be achieved through configuration.
- Search the existing [GitHub issues](https://github.com/duonl/synthpop-py/issues) to see whether the enhancement has already been suggested. If a similar proposal already exists, contribute to the existing discussion rather than opening a duplicate issue.
- Consider whether the proposal fits the scope and aims of the project. Keep in mind that we want features that will be useful to the majority of our users and not just a small subset. If you're just targeting a minority of users, consider writing an add-on/plugin library.

For larger changes, we recommend discussing the proposal before investing significant time in implementation. This helps ensure that the proposed approach aligns with the project's direction.

<!-- omit in toc -->
#### Submitting an enhancement suggestion

Enhancement suggestions are tracked through [GitHub issues](https://github.com/duonl/synthpop-py/issues).

A good enhancement request should include:
- a clear and descriptive title;
- a detailed (step-by-step) description of the proposed change;
- the current behaviour (where relevant) and the behaviour you would like to see;
- an explanation of why the change would be useful;
- any alternative approaches you have considered.

Where useful, you may also include screenshots, examples, references to similar implementations or other material that helps explain the proposal.

For the full process for designing and implementing a new feature, see the [develop documentation on defining a new feature](https://synthpop-py.readthedocs.io/en/develop/developer/way_of_working/Defining_a_new_feature.html).

<!-- You might want to create an issue template for enhancement suggestions that can be used as a guide and that defines the structure of the information to be included. If you do so, reference it here in the description. -->

### Contributing code

#### Setting up your development environment
synthpop-py uses [**Poetry**](https://python-poetry.org/docs/) to manage dependencies and the development environment.

We recommend installing Poetry using [**pipx**](https://pipx.pypa.io/stable/installation/#on-windows). You can check whether Poetry and pipx are installed by running the following commands in a terminal:
```bash
poetry --version
pipx --version
```

Next, clone the [repository](https://github.com/duonl/synthpop-py.git):
```bash
git clone https://github.com/duonl/synthpop-py.git
```

Change into the repository directory and install the project together with the development and documentation dependencies:
```bash
cd synthpop-py
poetry install --with docs,dev
```

#### Development workflow
The full development workflow, including guidance on implementing new functionality, is described in the [developer documentation](https://synthpop-py.readthedocs.io/en/develop/developer/way_of_working/developing.html).

### Improving the documentation
Documentation contributions are always welcome.

For small corrections or improvements, such as fixing spelling, grammar, examples, or unclear wording, you can generally open a pull request directly.

For larger changes, such as adding new guides or substantially restructuring existing documentation, we recommend discussing the proposed changes first. If the documentation change is connected to a new feature, please follow the project's workflow for developing new functionality.

## Style guides and conventions
Please follow the project's coding, documentation and developer conventions.

The relevant guidance can be found in the [style guides and conventions](https://synthpop-py.readthedocs.io/en/develop/developer/way_of_working/code_standards_and_norms.html).

The project also provides checklists for both development and code review:
- [Developer checklist](https://synthpop-py.readthedocs.io/en/develop/developer/way_of_working/checklist_for_developer.html)
- [Review checklist](https://synthpop-py.readthedocs.io/en/develop/developer/way_of_working/checklist_for_review.html)

## Thank you
Every contribution helps improve synthpop-py, whether it is a bug report, documentation correction, feature suggestion, or code contribution.

Thank you for taking the time to help improve the project.