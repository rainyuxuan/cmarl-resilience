import collections
import random


class TeamManager:

    def __init__(self, agents: list[str], my_team: str = None):
        self.agents = agents
        self.teams = self.group_agents()
        self.terminated_agents = set()
        self.my_team = my_team
        self.random_agents = None
        self.get_random_agents(1)

    def get_teams(self) -> list[str]:
        """
        Get the team names.
        :return: a list of team names
        """
        return list(self.teams.keys())

    def get_my_team(self):
        if self.my_team is not None:
            return self.my_team
        if 'tiger' in self.teams:
            my_team = 'tiger'
        elif 'predator' in self.teams:
            my_team = 'predator'
        else:
            my_team = self.get_teams()[0]
        self.my_team = my_team
        return my_team

    def get_team_agents(self, team: str) -> list[str]:
        """
        Get the agents in a team.
        :param team: the team name
        :return: a list of agent names in the team
        """
        assert team in self.teams, f"Team [{team}] not found."
        return self.teams[team]

    def get_my_agents(self) -> list[str]:
        return self.get_team_agents(self.get_my_team())

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

    def get_my_terminated_agents(self) -> list[str]:
        return list(self.terminated_agents.intersection(self.get_my_agents()))

    def get_random_agents(self, rate: float):
        """
        Create a random agent list, and return the first n agents.
        :param rate: the rate of random agents to return
        :return: a list of random agents with the length of rate * num_agents
        """
        num_agents = len(self.get_my_agents())
        if self.random_agents is not None:
            num_random_agents = int(num_agents * rate)
            return self.random_agents[:num_random_agents]
        else:
            self.random_agents = random.sample(self.get_my_agents(), num_agents)
            return self.get_random_agents(rate)

    @staticmethod
    def merge_terminates_truncates(terminates: dict[str, bool], truncates: dict[str, bool]) -> dict[str, bool]:
        """
        Merge terminates and truncates into one dictionary.
        :param terminates: a dictionary with agent names as keys and boolean values as values
        :param truncates: a dictionary with agent names as keys and boolean values as values
        :return: a dictionary with agent names as keys and boolean values as values
        """
        result = {}
        for agent in terminates:
            result[agent] = terminates[agent] or truncates[agent]
        return result