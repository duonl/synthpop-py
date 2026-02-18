# Checklist for review

## Before you review
- [ ] Understand the feature request
- [ ] Understand which user is going to use this.
- [ ] Understand how the user is going to use it.
- [ ] Understand the scope of the pull request. 
- [ ] Understand the issues related to this pull request. 
- [ ] Clarify the scope of the review.

The following sub list should be checked in order. If one sub list is not complete, request changes before reviewing any further. 
After changes have been submitted, start with the first sublist again. 

## 1. Global scan
- [ ] Only relevant files have been included in the pull request (so no automatically generated files or files that should not be public)
- [ ] You know what files you need to review. This means that you know what files have been reviewed already or are for someone else to review. 
- [ ] The pull request only includes the requested feature. 
- [ ] All merge conflicts have been resolved.
- [ ] If you pull the feature branch and run all the test, they all pass without warnings. 
- [ ] If you pretend to be an user and use this feature, it works as expected. Use the user examples in the issue.
- [ ] The documentations builds without errors. 


## 2. Reviewing the tests

- [ ] Every aspect of the new feature is being taken up in the unit tests. Standard aspects include:
    - correct shape and datatype of return value (correct dataframe/series, number of columns, datatype of columns, number of rows, column names). Typehints are non-binding. And a type hint might specify and abstract type, where the specification for a feature specify a concrete type.
    - behaviour in "trivial" cases (input is empty dataframe, zero, None, etc...)
    - behaviour in realistic use case. 
    - edge cases. (values equal to border values, None, etc...)
- [ ] unit tests are still lightweight (run quickly and do not use a lot of computer resources) and fully automatic. This is to enable automatic checks. Besides, unittests are supposed to run very often. Slow unit tests become annoying when developing, hindering testing more often.
For each unit test:
- [ ] The flow of the test is easy to understand. For each line of code of the test, you know what it does and it makes sense to have that line of code.
- [ ] The test leaves the computer the way it found it. The test should leave no proof that it has run. This implies that a cleanup should happen, even if the test fails. Check this by examining the code of the test. Are there files on disk being made? Are there any static or global variables. 
- [ ] It is clear what the test is supposed to test. A test has a certain claim,assertion or hypothesis over the code. It is clear what that hypothesis is. 
- [ ] The only reason the test could fail is an error in the code being tested (and not in dependencies)
- [ ] The tests effectively proves the claim.
- [ ] The test is not testing too much. A failed test should imply that the hypothesis should be rejected. So a bug in a dependency can not cause this test to fail. 

## 3. Reviewing the code

- [ ] The code is the most efficient way to pass the tests.
- [ ] There is no duplicate code.
- [ ] Each component (module, class, function) is easy to understand. 
- [ ] The SOLID principles have been applied. 
- [ ] Any edge cases you see in the code have been covered by unit tests. 
- [ ] Type hints where appropriate. 

## 4. Reviewing the form, style, and documentation

- [ ] PEP8 code standard.
- [ ] British english.
- [ ] The docstrings tell how to use the code. 
- [ ] There are examples of how to use the code, and those examples run. 