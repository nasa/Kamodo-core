# CHANGELOG.md

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/)

This project uses [calendar versioning](https://calver.org/)

## kamodo-22.5.0

### Added
- asyncio RPC interface
- dockerfile builds for py37 and py38
- docker compose for RPC test
- Partial keyword to the plotting call to make 2D or 3D plotting of higher dimension datasets simpler. 
- Example for using the gridify decorator.
- Contribution/Reporting/Support guidelines
- Description of how to run (or automate) the test suite.
- API documentation.
- Persistent links to JOSS paper references.
- Documentation of plot_partials plotting keyword.


### Changed
- Dependency list in the documentation.
- Joss Paper.
- Clarification of examples.
- Broken ink of sympy.
- Units in examples.

### Fixed
- Units missing on axes and color bars in kamodo plots.
- Default inheritance raises error for certain orderings.
- Argument units mixed with Dimensionless breaks unit conversion. 
- Error message for incompatible units.
- Set alphabetized ordering for arguments.
- Partial kwargs rendering.
- Incorrect axis labels for multi-variable plotting in 1D.
- User ordering with functions.
- Incorrect symbol registration.
- Function assignment changes function's repr_latex method.
- Function reassignment keeps old units.
- Field Integration notebooks problem.
- 3D Multi-variable plotting.
