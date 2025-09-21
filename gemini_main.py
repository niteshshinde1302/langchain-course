from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama

load_dotenv()

def main():
    print("Hello from langchain-course with Gemini!")
    information = """
    Warren Edward Buffett (/ˈbʌfɪt/ BUFF-it; born August 30, 1930)[2] is an American investor and philanthropist who currently serves as the chairman and CEO of the conglomerate holding company Berkshire Hathaway. As a result of his investment success, Buffett is one of the best-known investors in the world. According to Forbes, as of May 2025, Buffett's estimated net worth stood at US$160.2 billion, making him the fifth-richest individual in the world.[3]

Buffett was born in Omaha, Nebraska. The son of U.S. congressman and businessman Howard Buffett, he developed an interest in business and investing during his youth. He entered the Wharton School of the University of Pennsylvania in 1947 before graduating from the University of Nebraska in Lincoln at 20. He went on to graduate from Columbia Business School, where he molded his investment philosophy around the concept of value investing pioneered by Benjamin Graham. He attended New York Institute of Finance to focus on his economics background and soon pursued a business career.

He later began various business ventures and investment partnerships, including one with Graham. He created Buffett Partnership Ltd. in 1956 and his investment firm eventually acquired a textile manufacturing firm, Berkshire Hathaway, assuming its name to create a diversified holding company. Buffett emerged as the company's chairman and majority shareholder in 1970. In 1978, fellow investor and long-time business associate Charlie Munger joined Buffett as vice-chairman.[4][5]

Since 1970,[needs update] Buffett has presided as the chairman and largest shareholder of Berkshire Hathaway, one of America's foremost holding companies and world's leading corporate conglomerates. He has been referred to as the "Oracle" or "Sage" of Omaha by global media as a result of having accumulated a massive fortune derived from his business and investment success.[6][7] He is noted for his adherence to the principles of value investing, and his frugality despite his wealth.[8] Buffett has pledged to give away 99 percent[9] of his fortune to philanthropic causes, primarily via the Gates Foundation. He founded the Giving Pledge in 2010 with Bill Gates, whereby billionaires pledge to give away at least half of their fortunes.[10] At Berkshire Hathaway's investor conference on May 3, 2025, Buffett requested that the board appoint Greg Abel to succeed him as the company's chief executive officer by the year's end, whilst remaining chairman.[11] 
    """
    summary_template = f"""
    Given the information {information} about a person I want you to create:
    1. A short summary
    2. two interesting facts about them
    """

    summary_prompt_template = PromptTemplate(
        template=summary_template,
        input_variables=["information"]
    )

    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)
    #llm = ChatOllama(model="gemma3:270m", temperature=0)
    chain = summary_prompt_template | llm
    response = chain.invoke(
        input={"information": information}
    )
    print(response.content)

if __name__ == "__main__":
    main()
