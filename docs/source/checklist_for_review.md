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
- [ ] Only relevant files have been included in the pull request
- [ ] You know what files you need to review.
- [ ] The pull request only includes the requested feature. 
- [ ] All merge conflicts have been resolved.
- [ ] If you pull the feature branch and run all the test, they all pass without warnings. 

## 2. Reviewing the tests

- [ ] Every aspect of the new feature is being taken up in the unit tests. 
For each unit test:
- [ ] The test is easy to understand.
- [ ] It is clear what the test is supposed to test.
- [ ] The only reason the test could fail is an error in the code being tested.
- [ ] The tests effectively proves the claim.