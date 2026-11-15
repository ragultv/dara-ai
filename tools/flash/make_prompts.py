"""Training prompts for Dara Flash (the speculative drafter): a diverse, locally generated set of user requests in English and Tamil, combined from task
templates and topics. No external datasets: the drafter is trained by self-distillation on the target model's own answers to these prompts.

Usage: python -I tools/flash/make_prompts.py --n 2000 --out build/flash/prompts.jsonl [--seed 1]
"""
import argparse, json, os, random

TOPICS = [
    "solar panels for homes", "the history of the Chola dynasty", "learning to cook biryani", "a small bakery business", "monsoon flood preparedness",
    "electric scooters", "cricket training for teenagers", "remote work productivity", "saving money on groceries", "the water cycle", "photosynthesis",
    "machine learning basics", "a trip to Ooty", "Tamil Nadu temple architecture", "starting a YouTube channel", "first aid for burns", "time management for students",
    "renewable energy in India", "how vaccines work", "writing a resume", "job interview preparation", "public speaking", "a birthday party for a 7-year-old",
    "home gardening on a balcony", "the stock market for beginners", "climate change", "the human heart", "Chennai metro rail", "a wedding budget",
    "customer service for an online shop", "data privacy", "learning Python", "the French Revolution", "healthy breakfast ideas", "yoga for back pain",
    "a mobile app for farmers", "recycling plastic", "the solar system", "a road trip from Chennai to Pondicherry", "managing a team", "smartphone battery life",
    "the importance of sleep", "a library management system", "an e-commerce return policy", "a community clean-up drive", "teaching kids about money",
    "coffee brewing methods", "the Indian monsoon", "a hospital appointment system", "learning a new language", "cybersecurity at home", "a startup pitch",
    "electric vehicle charging", "rainwater harvesting", "a science fair project", "the benefits of reading", "a weekly meal plan", "a cloud migration project",
    "a quarterly sales report", "a new office policy", "a product launch", "a software bug report", "a customer complaint about late delivery", "a school annual day",
]
TEMPLATES = [
    "Write a short email to a colleague about {t}.", "Explain {t} in simple terms.", "Give me five practical tips about {t}.", "Write a brief summary of {t}.",
    "What are the pros and cons of {t}?", "Write a step-by-step plan for {t}.", "Write a short story involving {t}.", "Create a checklist for {t}.",
    "Answer as a teacher: what should a beginner know about {t}?", "Write a LinkedIn post about {t}.", "Write a polite reply to a customer asking about {t}.",
    "Compare two approaches to {t} in a short table.", "Write a three-paragraph blog introduction about {t}.", "List common mistakes people make with {t}.",
    "Draft meeting notes from a discussion about {t}, with action items.", "Write five frequently asked questions and answers about {t}.",
    "Describe {t} to a ten-year-old.", "Write a persuasive paragraph about {t}.", "Create a one-week schedule related to {t}.", "Write a formal letter about {t}.",
]
CODE = [
    "Write a Python function that {c}. Include a docstring.", "Write a JavaScript function that {c}.", "Explain how to {c} in Python with an example.",
    "Write a SQL query that {c2}.", "Write unit tests in pytest for a function that {c}.", "Refactor this idea into clean Python code: a function that {c}.",
]
CODE_TASKS = [
    "reverses the words in a sentence", "checks whether a string is a palindrome", "computes the median of a list", "removes duplicates from a list while keeping order",
    "parses a CSV line into fields", "counts word frequencies in a text", "validates an email address", "converts Celsius to Fahrenheit for a list of values",
    "finds the longest common prefix of strings", "merges two sorted lists", "flattens a nested list", "groups records by a key", "computes compound interest",
    "returns the n-th Fibonacci number iteratively", "checks if a number is prime", "formats a date as DD-MM-YYYY", "reads a JSON file and returns a dict",
]
SQL_TASKS = [
    "returns the top five customers by total spend", "counts orders per day for the last week", "finds employees without a manager",
    "lists products that were never ordered", "computes the average salary per department", "returns duplicate email addresses",
]
TAMIL = [
    "{t} பற்றி எளிய தமிழில் விளக்குக.", "{t} குறித்து ஐந்து பயனுள்ள குறிப்புகள் தருக.", "{t} பற்றி ஒரு சிறிய கட்டுரை எழுதுக.",
    "{t} குறித்து ஒரு நண்பருக்குக் கடிதம் எழுதுக.", "{t} பற்றிய நன்மைகள் மற்றும் தீமைகளைப் பட்டியலிடுக.",
]
TAMIL_TOPICS = [
    "மழைநீர் சேகரிப்பு", "சூரிய ஆற்றல்", "இணையப் பாதுகாப்பு", "நேர மேலாண்மை", "ஆரோக்கியமான உணவு", "திருக்குறள்", "சென்னை மெட்ரோ ரயில்",
    "விவசாயத்தில் தொழில்நுட்பம்", "மின்சார வாகனங்கள்", "யோகா", "புத்தக வாசிப்பு", "சுற்றுச்சூழல் பாதுகாப்பு", "சிறு தொழில் தொடங்குவது", "கல்வியில் கணினி",
]
EDIT_SNIPPETS = [
    ("Fix the grammar in this sentence: \"{s}\"", ["Me and him goes to the market every sundays.", "She don't like the movie that we watched yesterday.",
                                                   "There is many reasons why the project was delay.", "Their going to finish the work tomorow."]),
    ("Translate to Tamil: \"{s}\"", ["The meeting is postponed to Monday.", "Please send me the report by evening.", "Thank you for your quick response."]),
    ("Rewrite more formally: \"{s}\"", ["hey can u send the files asap", "the app is broken again, fix it", "we need more people on this, now"]),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default="build/flash/prompts.jsonl")
    a = ap.parse_args()
    rng = random.Random(a.seed)
    pool = set()
    for t in TOPICS:
        for tpl in TEMPLATES:
            pool.add(tpl.format(t=t))
    for c in CODE_TASKS:
        for tpl in CODE:
            if "{c}" in tpl:
                pool.add(tpl.format(c=c))
    for c2 in SQL_TASKS:
        pool.add(CODE[3].format(c2=c2))
    for t in TAMIL_TOPICS:
        for tpl in TAMIL:
            pool.add(tpl.format(t=t))
    for tpl, items in EDIT_SNIPPETS:
        for s in items:
            pool.add(tpl.format(s=s))
    prompts = sorted(pool)
    rng.shuffle(prompts)
    prompts = prompts[: a.n]
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        for i, p in enumerate(prompts):
            f.write(json.dumps({"id": i, "prompt": p}, ensure_ascii=False) + "\n")
    print(f"{len(prompts)} prompts from a pool of {len(pool)} -> {a.out}")


if __name__ == "__main__":
    main()
