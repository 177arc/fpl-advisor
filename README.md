[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/177arc/fpl-advisor/master?filepath=advisor.ipynb)
[![Python 3.6](https://img.shields.io/badge/python-3.6-blue.svg)](https://www.python.org/downloads/release/python-360/)

# Fantasy Premier League (FPL) Advisor

**WARNING: Playing FPL can be highly adictive.**

## Purpose
The purpose of the Advisor Jupyter notebook is to help with the selection of team members for the [Fantasy Premier League](https://fantasy.premierleague.com/) (FPL) by attempting to forecast how many points players will earn. It accesses the FPL API to download up-to-date stats, provides visual analysis and uses linear optimisation to recommend a team with the maximum expected points to improve the performance of your current team.

If you are not familar with the Fantasy Permier League, you can watch this introduction:

<a href="http://www.youtube.com/watch?v=SV_F-cL8fC0" target="_blank"><img src="http://img.youtube.com/vi/SV_F-cL8fC0/0.jpg" 
alt="How to play FPL" width="600" height="400"/></a>

Jupyter notebook for interactively analysing Fantasy Premier League (FPL) players and optimising your team's performance

## Usage

To use the Jupyter notebook interactively, simply open the [advisor.ipynb notebook on Binder](https://mybinder.org/v2/gh/177arc/fpl-advisor/master?filepath=advisor.ipynb) (it may take a bit of time to deploy the notebook). To simply view it, open the [advisor.ipynb notebook on NBViewer](https://nbviewer.jupyter.org/github/177arc/fpl-advisor/blob/master/advisor.ipynb).

Alternatively, simply clone the repository and open advisor.ipynb locally.

## Contributing

1. Fork the repository on GitHub.
2. Run the tests with `python -m pytest tests/` to confirm they all pass on your system.
   If the tests fail, then try and find out why this is happening. If you aren't
   able to do this yourself, then don't hesitate to either create an issue on
   GitHub, contact me on Discord or send an email to [py@177arc.net](mailto:py@177arc.net>).
3. Either create your feature and then write tests for it, or do this the other
   way around.
4. Run all tests again with with `python -m pytest tests/` to confirm that everything
   still passes, including your newly added test(s).
5. Create a pull request for the main repository's ``master`` branch.
