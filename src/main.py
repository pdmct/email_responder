from crewai import Crew, Agent, Task, Process
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_groq import ChatGroq
from datetime import datetime
from random import randint
from langchain.tools import tool
import json
import os

import requests
from langchain.tools import tool

import os
from crewai import Agent, Task, Crew, Process
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.agents import load_tools
from util import *

search_tool = DuckDuckGoSearchRun()

GROQ_LLM = ChatGroq(
            api_key=os.getenv("GROQ_API_KEY"),
            model="llama3-70b-8192"
)


EMAIL = """HI there, \n
I am emailing to say that the resort weather was way to cloudy and overcast. \n
I wanted to write a song called 'Here comes the sun but it never came'

What should be the weather in Arizona in April?

I really hope you fix this next time.

Thanks,
George
"""


class EmailAgents():
    def make_categorizer_agent(self):
        return Agent(
            role='Email Categorizer Agent',
            goal="""take in a email from a human that has emailed out company email address and categorize it \
            into one of the following categories: \
            price_equiry - used when someone is asking for information about pricing \
            customer_complaint - used when someone is complaining about something \
            product_enquiry - used when someone is asking for information about a product feature, benefit or service but not about pricing \\
            customer_feedback - used when someone is giving feedback about a product \
            off_topic when it doesnt relate to any other category """,
            backstory="""You are a master at understanding what a customer wants when they write an email and are able to categorize it in a useful way""",
            llm=GROQ_LLM,
            verbose=True,
            allow_delegation=False,
            max_iter=5,
            memory=True,
            step_callback=lambda x: print_agent_output(x,"Email Categorizer Agent"),
        )

    def make_researcher_agent(self):
        return Agent(
            role='Info Researcher Agent',
            goal="""take in a email from a human that has emailed out company email address and the category \
            that the categorizer agent gave it and decide what information you need to search for for the email writer to reply to \
            the email in a thoughtful and helpful way.
            If you DONT think a search will help just reply 'NO SEARCH NEEDED'
            If you dont find any useful info just reply 'NO USEFUL RESESARCH FOUND'
            otherwise reply with the info you found that is useful for the email writer
            """,
            backstory="""You are a master at understanding what information our email writer needs  to write a reply that \
            will help the customer""",
            llm=GROQ_LLM,
            verbose=True,
            max_iter=5,
            allow_delegation=False,
            memory=True,
            tools=[search_tool],
            step_callback=lambda x: print_agent_output(x,"Info Researcher Agent"),
        )

    def make_email_writer_agent(self):
        return Agent(
            role='Email Writer Agent',
            goal="""take in a email from a human that has emailed out company email address, the category \
            that the categorizer agent gave it and the research from the research agent and \
            write a helpful email in a thoughtful and friendly way.

            If the customer email is 'off_topic' then ask them questions to get more information.
            If the customer email is 'customer_complaint' then try to assure we value them and that we are addressing their issues.
            If the customer email is 'customer_feedback' then try to assure we value them and that we are addressing their issues.
            If the customer email is 'product_enquiry' then try to give them the info the researcher provided in a succinct and friendly way.
            If the customer email is 'price_equiry' then try to give the pricing info they requested.

            You never make up information. that hasn't been provided by the researcher or in the email.
            Always sign off the emails in appropriate manner and from Sarah the Resident Manager.
            """,
            backstory="""You are a master at synthesizing a variety of information and writing a helpful email \
            that will address the customer's issues and provide them with helpful information""",
            llm=GROQ_LLM,
            verbose=True,
            allow_delegation=False,
            max_iter=5,
            memory=True,
            step_callback=lambda x: print_agent_output(x,"Email Writer Agent"),
        )


class EmailTasks():
    # Define your tasks with descriptions and expected outputs
    def categorize_email(self, email_content):
        return Task(
            description=f"""Conduct a comprehensive analysis of the email provided and categorize into \
            one of the following categories:
            price_equiry - used when someone is asking for information about pricing \
            customer_complaint - used when someone is complaining about something \
            product_enquiry - used when someone is asking for information about a product feature, benefit or service but not about pricing \\
            customer_feedback - used when someone is giving feedback about a product \
            off_topic when it doesnt relate to any other category \

            EMAIL CONTENT:\n\n {email_content} \n\n
            Output a single cetgory only""",
            expected_output="""A single categtory for the type of email from the types ('price_equiry', 'customer_complaint', 'product_enquiry', 'customer_feedback', 'off_topic') \
            eg:
            'price_enquiry' \
            """,
            output_file=f"email_category.txt",
            agent=categorizer_agent
            )

    def research_info_for_email(self, email_content):
        return Task(
            description=f"""Conduct a comprehensive analysis of the email provided and the category \
            provided and search the web to find info needed to respond to the email

            EMAIL CONTENT:\n\n {email_content} \n\n
            Only provide the info needed DONT try to write the email""",
            expected_output="""A set of bullet points of useful info for the email writer \
            or clear instructions that no useful material was found.""",
            context = [categorize_email],
            output_file=f"research_info.txt",
            agent=researcher_agent
            )

    def draft_email(self, email_content):
        return Task(
            description=f"""Conduct a comprehensive analysis of the email provided, the category provided\
            and the info provided from the research specialist to write an email. \

            Write a simple, polite and too the point email which will respond to the customer's email. \
            If useful use the info provided from the research specialist in the email. \

            If no useful info was provided from the research specialist the answer politely but don't make up info. \

            EMAIL CONTENT:\n\n {email_content} \n\n
            Output a single cetgory only""",
            expected_output="""A well crafted email for the customer that addresses their issues and concerns""",
            context = [categorize_email, research_info_for_email],
            agent=email_writer_agent,
            output_file=f"draft_email.txt",
            )


##
agents = EmailAgents()
tasks = EmailTasks()

## Agents
categorizer_agent = agents.make_categorizer_agent()
researcher_agent = agents.make_researcher_agent()
email_writer_agent = agents.make_email_writer_agent()

## Tasks
categorize_email = tasks.categorize_email(EMAIL)
research_info_for_email = tasks.research_info_for_email(EMAIL)
draft_email = tasks.draft_email(EMAIL)


from crewai import Crew, Process

# Instantiate your crew with a sequential process
crew = Crew(
    agents=[categorizer_agent, researcher_agent, email_writer_agent],
    tasks=[categorize_email, research_info_for_email, draft_email],
    verbose=2,
    process=Process.sequential,
    full_output=True,
    share_crew=False,
    step_callback=lambda x: print_agent_output(x,"MasterCrew Agent")
)


# Kick off the crew's work
results = crew.kickoff()


# Print the results
print("Crew Work Results:")
print(results)

# print(f"Categorize Email: {categorize_email.output}")
print(crew.usage_metrics)
