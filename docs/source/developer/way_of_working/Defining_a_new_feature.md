# Defining a new feature

This guide describes the process of defining a feature, from initial idea to the start of development.

We welcome ideas for new features and improvements. If you have an idea that could make synthpop-py more useful, please get in touch with us by opening an issue on [GitHub](https://github.com/duonl/synthpop-py/issues/new). You do not need to have a complete design, implementation plan, or code ready before starting the conversation.

## What kind of new features we are looking for
The aim of this package is to facilitate the creation of synthetic data. The primary intended audience is government institutions.
For this reason, reliability and correctness are more important than performance. We cannot accept untested and/or unproven technology where doing so could compromise the reliability of the package.

At the same time, new features that improve usability are very welcome. We are also interested in features that address common user needs, support important use cases, or make it easier to work with synthetic data.

More features do not always make for a better package. Adding a new feature has several consequences:
- users need to understand the feature;
- it must be maintained; and
- it may introduce bugs or other complexity.

We therefore consider not only whether a feature is technically possible, but also whether it provides sufficient value to users and fits the goals and direction of the project.

## Step 1: Feature request
The first step is to let us know that you have an idea for an improvement. **Please do not hesitate to get in touch, even if your idea is still at an early stage.**

This can be done by submitting a feature request via [GitHub](https://github.com/duonl/synthpop-py/issues/new). You do not need to provide code or a detailed plan at this stage. The purpose of this step is to start a discussion and establish whether, and roughly when, we want this functionality in the package.

Some important things to mention in the initial feature request are:
- What problem does this solve?
- Who will use it?
- To what extent is this a proven concept? A proof of concept or additional information may be helpful at this stage.

A feature request does not need to answer all of these questions perfectly. The initial discussion is an opportunity for us to explore the idea together and determine whether it fits the project.

In the discussion that follows, the following points may be addressed:
- How many people will use this feature? Does it generalise to other user groups?
- What is the impact on users who benefit from this feature?
- What is the impact on other users?
- Who do you expect will develop this feature?

During this discussion, decisions will be made regarding the required functionality.
These decisions are consolidated into a document called the functional description.

The functional description defines the requirements for the feature and should remain independent of any specific programming language or implementation details.
This document serves as the basis for the next phase.

Based on this and the follow-up discussion, the product owner decides whether the feature will be included into the package.

If the feature is accepted, it moves on to the next step.

## Step 2: Design consensus
What exactly happens in this step depends on the feature.
The aim of this step is to reach consensus on how the feature should be implemented.
While it may be tempting to skip this step and immediately begin development, there is a risk that the feature will need to be completely rewritten to fit within the package.

Some things to establish in this step are:
- expected usage (for example, sample code showing how to use the feature)
- suggested code structure or mock-up (empty classes, methods and functions), including function signatures. Adding documentation to the mock-up can be very helpful in clarifying the plan
- additional dependencies. We need to be careful when introducing external packages, as they may cause compatibility or versioning conflicts.
- accepted user inputs
- (if applicable) notable edge cases
- which parts of the codebase need to change
- related development work. Other features may be under development that overlap with or depend on this feature.

There are multiple effective ways to communicate this. Describing it in natural language in the feature request may be a good starting point.
As the plan becomes more concrete, it may be helpful to implement a code mock-up in a branch. This can provide a head start for development.
By the end of this stage, a developer should have all the information needed to implement the feature.

## Step 3: Planning
The next step is to decide when development should begin. The product owner is responsible for this planning.

The following points are taken into account when scheduling development of a new feature:
- A release containing breaking changes (changes that require users to alter their code) places a burden on users. It may be best to bundle breaking changes into a single major release.
- A feature may have synergies with other features
- Developer capacity
- Urgency of the feature
- Overall priorities

## Have an idea?
If you have an idea for a new feature or an improvement to synthpop-py, **we would be happy to hear from you**. You do not need to have the implementation figured out before starting a discussion. You can find us on [GitHub](https://github.com/duonl/synthpop-py).

Starting with a feature request allows us to discuss the idea, provide feedback, and work out together whether and how it could become part of synthpop-py.

