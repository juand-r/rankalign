"""
One-time script: add generator_sentence and discriminator_sentence fields
to rosch-1975.json and incorrect_items-to-supplement-rosch-75.json.

After running, review the outputs and hand-fix any grammar issues.
"""

import json
from pathlib import Path

FLORA_DIR = Path(__file__).resolve().parent.parent.parent / "flora"
ROSCH_FILE = FLORA_DIR / "rosch-1975.json"
SUPPLEMENT_FILE = FLORA_DIR / "incorrect_items-to-supplement-rosch-75.json"

UNCOUNTABLE_CATEGORIES = {'Furniture', 'Clothing'}

# Three-way classification for members:
# 'plural' -> no article, "are"
# 'no_article' -> no article, "is"
# everything else -> singular, gets "a/an", "is"
PLURAL_MEMBERS = {
    'pants', 'shoes', 'boots', 'sandals', 'slippers', 'socks', 'stockings',
    'gloves', 'mittens', 'nylons', 'overshoes', 'pajamas', 'panties',
    'slacks', 'underpants', 'earmuffs', 'earrings', 'cuff links',
    'grapes', 'prunes', 'preserves',
    'drapes', 'blinds', 'drawers',
    'artichokes', 'beets', 'onions', 'peppers', 'pickles', 'radishes',
    'yams', 'greens', 'green beans', 'lima beans', 'string beans',
    'wax beans', 'baked beans', 'blackeyed peas', 'brussels sprouts',
    'green peppers', 'turnip greens',
    'fists', 'bricks', 'words',
    'nails', 'screws', 'bolts', 'nuts', 'pliers', 'rags', 'scissors',
    'blueprints', 'shavings',
    'marbles', 'jacks', 'skates', 'stilts', 'blocks', 'cards',
    'crayons', 'dishes', 'books', 'animals', 'paper dolls',
    'checkers', 'geese',
    'horseshoes',
}

NO_ARTICLE_MEMBERS = {
    'billiards', 'gymnastics',
    'goosebumps', "crow's feet", 'brass knuckles',
    'chess',
}

FORCE_AN = {'a', 'e', 'i', 'o', 'u'}


def get_article(word):
    if word[0].lower() in FORCE_AN:
        return 'an'
    return 'a'


def cat_phrase(category):
    """'furniture' or 'a bird'."""
    if category in UNCOUNTABLE_CATEGORIES:
        return category.lower()
    art = get_article(category.lower())
    return f"{art} {category.lower()}"


def gen_sentence(category, member):
    cp = cat_phrase(category)
    m_lower = member.lower()
    if m_lower in PLURAL_MEMBERS or m_lower in NO_ARTICLE_MEMBERS:
        return f"Complete the sentence: an example of {cp} is"
    art = get_article(member)
    return f"Complete the sentence: an example of {cp} is {art}"


def disc_sentence(category, member):
    cp = cat_phrase(category)
    m_lower = member.lower()
    if m_lower in PLURAL_MEMBERS:
        return f"Do you think {member} are {cp}?"
    elif m_lower in NO_ARTICLE_MEMBERS:
        return f"Do you think {member} is {cp}?"
    else:
        art = get_article(member)
        return f"Do you think {art} {member} is {cp}?"


def process_rosch():
    with open(ROSCH_FILE) as f:
        data = json.load(f)

    for cat, items in data['categories'].items():
        for item in items:
            item['generator_sentence'] = gen_sentence(cat, item['member'])
            item['discriminator_sentence'] = disc_sentence(cat, item['member'])

    with open(ROSCH_FILE, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"Updated {ROSCH_FILE}")


def process_supplement():
    with open(SUPPLEMENT_FILE) as f:
        data = json.load(f)

    for cat, items in data['categories'].items():
        for item in items:
            item['generator_sentence'] = gen_sentence(cat, item['member'])
            item['discriminator_sentence'] = disc_sentence(cat, item['member'])

    with open(SUPPLEMENT_FILE, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"Updated {SUPPLEMENT_FILE}")


if __name__ == '__main__':
    process_rosch()
    process_supplement()
    print("Done. Review the JSONs and fix any grammar issues.")
