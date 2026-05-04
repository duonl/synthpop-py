# Defining a new feature

This guide describes the process of defining a feature: from initial idea to the start of development.

## What kind of new features are we looking for
The aim of this package is to facilitate the creation of synthetic data. The first intended audience are goverment institutions.
Being reliable and trusthworthiness are more important than preformance. See also CONTRIBUTING.md. We cannot accept untested and/or unproven technology. 
On the other hand, new features that improve user friendliness are very welcome. 

More features does not always make for a better package. Adding a new feature has as a consequence that:
- users need to understand this feature
- it needs to be maintained
- it is something that might contain bugs. 

## Step 1: a feature request
The first step is to let us know that you have an idea for an improvement.This can be done by submitting a feature request via GitHub. You don't need to provide code or a detailed plan yet.
The purpose of this is to establish if (and roughly when) we want this functionality in this package. 
Some important things to mention in the initial feature request are:
- What problem does this solve?
- Who is going to use this?
- To what extend is this a proven concept? This is the point where a proof of concept might help. Is there other info

In the discussion that follow, there can be discussion about:
- How many people are going to use this? Does this generalize to other groups of users?
- What is the impact on users that profit from this feature?
- What is the impact on the other users?
- Who do you hope/expect will develop this?
Based on this and the follow-up discussion, the product owner decides if this is a feature that will be adopted in this package.
If the feature gets accepted, it moves on to the next step

## Step 2: Design consensus
What exactly happens in this step depends mostly on the feature. 
The aim of this step is to reach consensus on how the feature should be implemented. 
While it might be tempting to skip this step and immediatly develop it, there is a risk that it needs to be completely rewritten to fit in with this package.
Some things to establish in this step are:
- expected usage (example code of how to use this feature)
- suggested code layout/ mockup of the code (empty classes, empty methods and functions), function signatures. Adding documentation to the mockup can help a lot to understand the plan.
- extra dependencies
- accepted user inputs
- (if applicable) notable edge cases
- which parts of the codebase need to change
- related development
There are multiple effective ways to comunicate this. Describing it in natural language in the feature request might be a good point to start.
As the plans become less abstract, it might help to write the mockup code in a branch. This could give development a head start.
At the end of this stage, a developer would have all the information they need to develop this feature.

## Step 3: planning
The next step is to decide when to develop this feature. The product owner is responsible for this planning.
The following points are taken into account when planning the development of a new feature:
- A release that has breaking changes (changes that force users to alter their code) places a burden on users. It might be best to bundle breaking changes into one major version release.
- A feature might have synergy with other features.
- capacity of developers
- urgency of the feature
- priorities


