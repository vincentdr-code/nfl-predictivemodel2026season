"""
NFL team metadata: names, colors, logos, division alignment.

Logo URLs use ESPN's public CDN — no key required.
Colors are official brand primaries as of the 2026 season.
"""

TEAMS = {
    'ARI': {'name': 'Cardinals',   'city': 'Arizona',       'conf': 'NFC', 'div': 'West',  'primary': '#97233F', 'secondary': '#000000'},
    'ATL': {'name': 'Falcons',     'city': 'Atlanta',       'conf': 'NFC', 'div': 'South', 'primary': '#A71930', 'secondary': '#000000'},
    'BAL': {'name': 'Ravens',      'city': 'Baltimore',     'conf': 'AFC', 'div': 'North', 'primary': '#241773', 'secondary': '#9E7C0C'},
    'BUF': {'name': 'Bills',       'city': 'Buffalo',       'conf': 'AFC', 'div': 'East',  'primary': '#00338D', 'secondary': '#C60C30'},
    'CAR': {'name': 'Panthers',    'city': 'Carolina',      'conf': 'NFC', 'div': 'South', 'primary': '#0085CA', 'secondary': '#101820'},
    'CHI': {'name': 'Bears',       'city': 'Chicago',       'conf': 'NFC', 'div': 'North', 'primary': '#0B162A', 'secondary': '#C83803'},
    'CIN': {'name': 'Bengals',     'city': 'Cincinnati',    'conf': 'AFC', 'div': 'North', 'primary': '#FB4F14', 'secondary': '#000000'},
    'CLE': {'name': 'Browns',      'city': 'Cleveland',     'conf': 'AFC', 'div': 'North', 'primary': '#311D00', 'secondary': '#FF3C00'},
    'DAL': {'name': 'Cowboys',     'city': 'Dallas',        'conf': 'NFC', 'div': 'East',  'primary': '#003594', 'secondary': '#869397'},
    'DEN': {'name': 'Broncos',     'city': 'Denver',        'conf': 'AFC', 'div': 'West',  'primary': '#FB4F14', 'secondary': '#002244'},
    'DET': {'name': 'Lions',       'city': 'Detroit',       'conf': 'NFC', 'div': 'North', 'primary': '#0076B6', 'secondary': '#B0B7BC'},
    'GB':  {'name': 'Packers',     'city': 'Green Bay',     'conf': 'NFC', 'div': 'North', 'primary': '#203731', 'secondary': '#FFB612'},
    'HOU': {'name': 'Texans',      'city': 'Houston',       'conf': 'AFC', 'div': 'South', 'primary': '#03202F', 'secondary': '#A71930'},
    'IND': {'name': 'Colts',       'city': 'Indianapolis',  'conf': 'AFC', 'div': 'South', 'primary': '#002C5F', 'secondary': '#A2AAAD'},
    'JAX': {'name': 'Jaguars',     'city': 'Jacksonville',  'conf': 'AFC', 'div': 'South', 'primary': '#101820', 'secondary': '#D7A22A'},
    'KC':  {'name': 'Chiefs',      'city': 'Kansas City',   'conf': 'AFC', 'div': 'West',  'primary': '#E31837', 'secondary': '#FFB81C'},
    'LA':  {'name': 'Rams',        'city': 'Los Angeles',   'conf': 'NFC', 'div': 'West',  'primary': '#003594', 'secondary': '#FFA300'},
    'LAC': {'name': 'Chargers',    'city': 'Los Angeles',   'conf': 'AFC', 'div': 'West',  'primary': '#0080C6', 'secondary': '#FFC20E'},
    'LAR': {'name': 'Rams',        'city': 'Los Angeles',   'conf': 'NFC', 'div': 'West',  'primary': '#003594', 'secondary': '#FFA300'},
    'LV':  {'name': 'Raiders',     'city': 'Las Vegas',     'conf': 'AFC', 'div': 'West',  'primary': '#000000', 'secondary': '#A5ACAF'},
    'MIA': {'name': 'Dolphins',    'city': 'Miami',         'conf': 'AFC', 'div': 'East',  'primary': '#008E97', 'secondary': '#FC4C02'},
    'MIN': {'name': 'Vikings',     'city': 'Minnesota',     'conf': 'NFC', 'div': 'North', 'primary': '#4F2683', 'secondary': '#FFC62F'},
    'NE':  {'name': 'Patriots',    'city': 'New England',   'conf': 'AFC', 'div': 'East',  'primary': '#002244', 'secondary': '#C60C30'},
    'NO':  {'name': 'Saints',      'city': 'New Orleans',   'conf': 'NFC', 'div': 'South', 'primary': '#D3BC8D', 'secondary': '#101820'},
    'NYG': {'name': 'Giants',      'city': 'New York',      'conf': 'NFC', 'div': 'East',  'primary': '#0B2265', 'secondary': '#A71930'},
    'NYJ': {'name': 'Jets',        'city': 'New York',      'conf': 'AFC', 'div': 'East',  'primary': '#125740', 'secondary': '#FFFFFF'},
    'OAK': {'name': 'Raiders',     'city': 'Oakland',       'conf': 'AFC', 'div': 'West',  'primary': '#000000', 'secondary': '#A5ACAF'},
    'PHI': {'name': 'Eagles',      'city': 'Philadelphia',  'conf': 'NFC', 'div': 'East',  'primary': '#004C54', 'secondary': '#A5ACAF'},
    'PIT': {'name': 'Steelers',    'city': 'Pittsburgh',    'conf': 'AFC', 'div': 'North', 'primary': '#101820', 'secondary': '#FFB612'},
    'SD':  {'name': 'Chargers',    'city': 'San Diego',     'conf': 'AFC', 'div': 'West',  'primary': '#0080C6', 'secondary': '#FFC20E'},
    'SEA': {'name': 'Seahawks',    'city': 'Seattle',       'conf': 'NFC', 'div': 'West',  'primary': '#002244', 'secondary': '#69BE28'},
    'SF':  {'name': '49ers',       'city': 'San Francisco', 'conf': 'NFC', 'div': 'West',  'primary': '#AA0000', 'secondary': '#B3995D'},
    'STL': {'name': 'Rams',        'city': 'St. Louis',     'conf': 'NFC', 'div': 'West',  'primary': '#003594', 'secondary': '#FFA300'},
    'TB':  {'name': 'Buccaneers',  'city': 'Tampa Bay',     'conf': 'NFC', 'div': 'South', 'primary': '#D50A0A', 'secondary': '#0A0A08'},
    'TEN': {'name': 'Titans',      'city': 'Tennessee',     'conf': 'AFC', 'div': 'South', 'primary': '#0C2340', 'secondary': '#4B92DB'},
    'WAS': {'name': 'Commanders',  'city': 'Washington',    'conf': 'NFC', 'div': 'East',  'primary': '#5A1414', 'secondary': '#FFB612'},
}

# ESPN CDN team code overrides where they differ from ours
ESPN_CODE_OVERRIDES = {
    'LA': 'lar',   # ESPN uses 'lar' for the current LA Rams
    'JAX': 'jax',
    'WAS': 'wsh',  # ESPN uses 'wsh' historically
    'OAK': 'lv',
    'SD':  'lac',
    'STL': 'lar',
}


def logo_url(team_code):
    """Return the ESPN CDN URL for a team's logo (500x500 PNG)."""
    code = ESPN_CODE_OVERRIDES.get(team_code, team_code.lower())
    return f"https://a.espncdn.com/i/teamlogos/nfl/500/{code}.png"


def team_name(code):
    """'BUF' -> 'Buffalo Bills'."""
    t = TEAMS.get(code)
    return f"{t['city']} {t['name']}" if t else code


def short_name(code):
    """'BUF' -> 'Bills'."""
    t = TEAMS.get(code)
    return t['name'] if t else code


def primary_color(code):
    return TEAMS.get(code, {}).get('primary', '#333333')


def secondary_color(code):
    return TEAMS.get(code, {}).get('secondary', '#666666')
