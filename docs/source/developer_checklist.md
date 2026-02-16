# Checklist for developers

This document specifies the tasks of the developer. 
This checklist is meant to ensure code quality and optimal code workflow. 

## Before you begin
- [ ] Fully understand the feature request. 
- [ ] Fully understand the proof of concept code. 
- [ ] Fully understand how this feature should be implemented
- [ ] Think of any edge cases or exceptions.

## During development

- [ ] Work on a branch based on the develop branch.
- [ ] Only implement the feature on this branch. Found other things that need to be done first? Make a different branch.
- [ ] Apply Test Driven Development. 
- [ ] Use the GIVEN, WHEN, THEN format for unit tests. Specify the GIVEN, WHEN, THEN in the docstring of the unit tests.
- [ ] Apply the PEP8 standard when useful. 
- [ ] Use British english.

## Before making a pull request

- [ ] You can demonstrate that it works on your machine.
- [ ] It works on your machine like it would on a users machine.
- [ ] All tests ar passing.
- [ ] Warnings are resolved. 
- [ ] The in-code docstrings accurately describe the methods and classes and how to use it.
- [ ] Any other relevant documentation has been updated. 
- [ ] Any irrelevant files are excluded from the branch (untrack/deleted).

## After making a pull request, before asking review
- [ ] The pull requests merges into develop.
- [ ] The title of the pull requests is descriptive and reflects issue numbers.
- [ ] The description of the pull requests gives a short summary what has been changed and highlights areas that need more attention (if applicable).
- [ ] Any merge conflicts have been resolved.
- [ ] All previous checks need to be rechecked after resolving merge conflicts. 

## After processing feedback
- [ ] All checks of "Before making a pull request" still apply after processing feedback.
- [ ] Any merge conflicts have been resolved.
- [ ] All previous checks need to be rechecked after resolving merge conflicts. 

You are done when the reviewer has no more feedback on the most recent state of the feature branch. 


