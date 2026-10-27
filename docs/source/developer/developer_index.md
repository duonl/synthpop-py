# For developers

Welcome to the developer documentation for synthpop-py!

This documentation is intended for developers working on the synthpop-py project. It provides a structured guide to the development process, the architecture of the project and the quality standards expected within the codebase.

Whether you are making your first contribution, developing a new feature, reviewing a pull request, or simply interested in understanding how synthpop-py works, these pages should help you find your way around the project.

If you are new to the project, we recommend starting with [Developing synthpop-py](way_of_working/developing.md). This provides the essential context: the overall principles, workflows, branching strategy, and how features are defined, developed, and released. It gives you the conceptual foundation needed to understand how work is organised.

Once familiar with the general workflow, use the [Checklist for Developers](way_of_working/checklist_for_developer.md) as a practical step-by-step guide during implementation. It outlines what to do before, during, and after development, including preparing a pull request and handling feedback. This helps us keep contributions consistent and complete.

When reviewing code, or preparing your own work for review, refer to the [Checklist for Review](way_of_working/checklist_for_review.md). This document structures the review process, covering global checks, tests, code quality, and documentation, and helps us maintain a high standard across contributions.

For questions about what constitutes acceptable code, consult [Code Standards, Norms, and Conventions](way_of_working/code_standards_and_norms.md). This document defines the criteria for code quality and consistency, and acts as a tie-breaker when multiple valid approaches are possible.

Finally, the [Dataflow Diagrams](way_of_working/dataflowdiagram.md) provide a visual and conceptual overview of how synthpop-py operates internally. Use them as a reference when you need to understand the system's architecture or the flow of data through its different components.

## Want to contribute?
We welcome contributions of all kinds. You can contribute by implementing a feature, fixing a bug, improving documentation, reviewing code, suggesting an idea, or helping us improve the development process.

You do not need to know the entire codebase before getting involved. If you have an idea for an improvement or are unsure where to start, **please get in touch with us**. Questions and discussions are welcome and we are happy to help you find a way to contribute.

The [Contributing to the package](../../../CONTRIBUTING.md) page provides more information about contributing to synthpop-py, including how to suggest enhancements, report bugs, improve documentation and contribute code.

If there is something you would like synthpop-py to provide, let us know! Your feedback helps us understand what is useful to the community and helps shape the future development of the project.

```{toctree}
:maxdepth: 1
:caption: Way of Working

way_of_working/developing.md
way_of_working/Defining_a_new_feature.md
way_of_working/checklist_for_developer.md
way_of_working/checklist_for_review.md
way_of_working/code_standards_and_norms.md
way_of_working/dataflowdiagram.md
way_of_working/randomness.md

```

```{toctree}
:maxdepth: 1
:caption: Functional Descriptions

Synthesis workflow <functional_descriptions/index_synthesis_workflow.md>
Data preparation <functional_descriptions/index_data_preparation.md>
Synthesis methods <functional_descriptions/index_synthesis_methods.md>
Evaluation <functional_descriptions/index_evaluation.md>
Supporting functionality <functional_descriptions/index_supporting_functionality.md>

```


```{toctree}
:maxdepth: 1
:caption: Documentation
documentation
```

```{toctree}
:maxdepth: 1
:caption: Contributing
contributing_to_package
```
