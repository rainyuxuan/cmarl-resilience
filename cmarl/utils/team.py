import collections


class TeamManager:

    def __init__(self, agents: list[str]):
        self.agents = agents
        self.teams = self.group_agents()
        self.terminated_agents = set()

    def get_teams(self) -> list[str]:
        """
        Get the team names.
        :return: a list of team names
        """
        return list(self.teams.keys())

    def get_my_team(self):
        if 'tiger' in self.teams:
            return 'tiger'
        elif 'predator' in self.teams:
            return 'predator'
        else:
            return self.get_teams()[0]

    def get_team_agents(self, team: str) -> list[str]:
        """
        Get the agents in a team.
        :param team: the team name
        :return: a list of agent names in the team
        """
        assert team in self.teams, f"Team [{team}] not found."
        return self.teams[team]

    def group_agents(self) -> dict[str, list[str]]:
        """
        Group agents by their team.
        :param agents: a list of agent names in the format of teamname_agentid
        :return: a dictionary with team names as keys and a list of agent names as values
        """
        teams = collections.defaultdict(list)
        for agent in self.agents:
            team, _ = agent.split('_')
            teams[team].append(agent)
        return teams

    def get_info_of_team(self, team: str, data: dict[str, any], default=None) -> dict[str, any]:
        """
        Get the information of a team.
        :param team: the team name
        :param data: the data to get information from
        :return: a dictionary with the team name as key and the information as value
        """
        assert team in self.teams, f"Team [{team}] not found."
        result = {}
        for agent in self.get_team_agents(team):
            if agent not in data:
                result[agent] = default
            else:
                result[agent] = data[agent]
        return result
    
    def reset(self):
        self.terminated_agents = set()

    def is_team_terminated(self, team: str):
        """
        Check if all agents in a team are terminated.
        :param team: the team name
        :return: True if all agents in the team are terminated, False otherwise
        """
        assert team in self.teams, f"Team [{team}] not found."
        return all(agent in self.terminated_agents for agent in self.teams[team])

    def terminate_agent(self, agent: str):
        """
        Mark an agent as terminated.
        :param agent:
        :return:
        """
        self.terminated_agents.add(agent)

    def has_terminated_teams(self) -> bool:
        """
        Check if any team is terminated.
        """
        for team in self.teams:
            if self.is_team_terminated(team):
                return True
        return False