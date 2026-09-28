# For developers

This documentation is intended for developers working on the Synthpop project. It serves as a structured guide to both the development process and the quality standards expected within the codebase.

If you are new to the project, it is recommended to start with [Developing Synthpop](way_of_working/developing.md), which provides the essential context: the overall principles, workflows, branching strategy, and how features are defined, developed, and released. This page gives you the conceptual foundation needed to understand how work is organised.

Once familiar with the general workflow, you should use the [Checklist for Developers](way_of_working/checklist_for_developer.md) as a practical step-by-step guide during implementation. It outlines what to do before, during, and after development, including preparing a pull request and handling feedback. This ensures consistency and completeness.

When reviewing code (or preparing your work for review), refer to the [Checklist for Review](way_of_working/checklist_for_review.md). This document structures the review process, covering global checks, tests, code quality, and documentation, helping maintain a high standard across contributions.

For questions about what constitutes acceptable code, consult [Code Standards, Norms, and Conventions](way_of_working/code_standards_and_norms.md). This document defines the criteria for code quality and consistency, and acts as a tie-breaker when multiple valid approaches exist.

Finally, the [Dataflow Diagrams](way_of_working/dataflowdiagram.md) provide a visual and conceptual overview of how Synthpop operates internally. Use this as a reference when you need to understand the system's architecture or the flow of data through different components.

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

Index <functional_descriptions/index_functional_descriptions.md>
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
contributing_placeholder
```
