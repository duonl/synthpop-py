# Checklist for developers

This document specifies the tasks of the developer. 
This checklist is meant to ensure code quality and optimal code workflow. 

## Before you begin
- [ ] Fully understand the feature request. 
- [ ] Fully understand the proof of concept code. 
- [ ] Fully understand how this feature should be implemented. Read the functional descriptions.
- [ ] Think of any edge cases or exceptions.

## During development

- [ ] Work on a branch based on the develop branch.
- [ ] Only implement the feature on this branch. Found other things that need to be done first? Open a new issue and make a different branch.
- [ ] Apply Test Driven Development. 
- [ ] Use the [GIVEN, WHEN, THEN format](https://martinfowler.com/bliki/GivenWhenThen.html) for unit tests. Specify the GIVEN, WHEN, THEN in the docstring of the unit tests.
- [ ] Apply the PEP8 standard when useful. 
- [ ] Use British English.

## Before making a pull request

- [ ] You can demonstrate that it works on your machine.
- [ ] It works on your machine like it would on a users machine.
- [ ] All tests are passing.
- [ ] Warnings are resolved. 
- [ ] The in-code docstrings accurately describe the methods and classes and how to use them.
- [ ] Any other relevant documentation has been updated. 
- [ ] Any irrelevant files are excluded from the branch (untrack/deleted).

## After making a pull request, before asking review
- [ ] The pull request merges into develop.
- [ ] The title of the pull request is descriptive and reflects issue numbers.
- [ ] The description of the pull request gives a short summary what has been changed and highlights areas that need more attention (if applicable). It also includes a description of the scope and intended user
Or just a link to the right issues
- [ ] Any merge conflicts have been resolved.
- [ ] All previous checks need to be rechecked after resolving merge conflicts. 

## After processing feedback
- [ ] All checks of "Before making a pull request" still apply after processing feedback.
- [ ] Any merge conflicts have been resolved.
- [ ] All previous checks need to be rechecked after resolving merge conflicts. 

You are done when the reviewer has no more feedback on the most recent state of the feature branch. After approval from a reviewer with write access, the pull request can be merged into develop. **Make sure the commits are first squashed before it is merged.** This can be done by clicking on *Squash and merge* on Github at the bottom of the pull request. Make sure to click on the dropdown menu if you only see a button with *Merge pull request*. To squash merge in the command line you first run `git checkout target_branch` to make sure you are on the branch you want to merge into. Then perform the squash merge by running `git merge --squash feature_branch` and finally commit the changes `git commit -m "Commit message"`.


