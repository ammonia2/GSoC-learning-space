# Motivation

## Who I am
<!-- Your background — what you study/work on, your programming experience, anything relevant. -->
I am a Bachelors Student studying Artificial Intelligence at NUCES FAST, Islamabad. Throughout my university and even through High School, I have extensively worked on Python programs (making games like scrabble, and more recently working on a language Interpreter).
I wanted to pick a domain in this vast expanse of AI and reinforcement learning is something that always captivated me, mainly because I saw RL Agents in popular games like Rocket League and I decided that I want to build something similar of my own. Over the past year, I have been learning about RL through David Silver's lectures, Richard Sutton's books, and papers on (MARL) Multi-Agent Reinforcement Learning algorithms (MAPPO, MADDPG, QMIX etc.) for my semester project in MARL Opponent Modeling.

## Why Mesa
<!-- What drew you to Mesa specifically? Have you used it before? How did you find it? -->
I found the organisation through GSoC. Using Mesa I found that I had a lot of control over everything (something you don't get in Blackbox RL Policy Gradients). Building SARSA football, I directly learned how small reward shaping tuned the agent scoring behaviours. That interpretability is what drew me in.

## What I want to learn
<!-- What aspects of Mesa or ABM are you most interested in? What skills do you want to develop? -->
My football model had abstractions including implicit behavioural states that are not visualised or tracked, step function simulating multiple actions (passing and tackling bundled into a single action) etc. I want to learn how to build the abstractions that are general enough to support BDI, needs-based and SARSA alike models while not being over done.
I want to work on the Action Framework and State Management components from [Mesa discussion 2538](https://github.com/mesa/mesa/discussions/2538). The design question that interests me is precisely the one EwoutH and tpike3 were debating, what is the minimal "genetic code" that lets users instantiate a wide range of behavioural theories without Mesa becoming another unmaintained side project?

## Where I want to go
<!-- What's your goal with contributing to Mesa? What kind of contributions do you see yourself making? -->
The challenges I faced are precisely what were mentioned in [discussion 2538](https://github.com/mesa/mesa/discussions/2538). Keeping track of agent behaviours and seeing how rewards shape decisions are the challenges that I faced building my football model. I understand that while working on these extensions, it is very important to also keep it simple so that the control is in the hand of the users. I want to contribute the Action Framework and State Management components, so that users can implement complex behavioral theories without rewriting the same scaffolding every time.