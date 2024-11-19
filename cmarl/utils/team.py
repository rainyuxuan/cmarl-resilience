
def group_agents(agents: list[str]) -> dict[str, list[str]]:
    """
    Group agents by their team.
    :param agents: a list of agent names in the format of teamname_agentid
    :return: a dictionary with team names as keys and a list of agent names as values
    """
    teams = {}
    for agent in agents:
        team, _ = agent.split('_')
        if team in teams:
            teams[team].append(agent)
        else:
            teams[team] = [agent]
    return teams

