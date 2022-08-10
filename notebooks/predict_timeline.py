import pickle
import pandas as pd
import yaml
from pathlib import Path
from dataclasses import dataclass
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


@dataclass
class Team:
    name: str
    score: int
    attack: int
    defense: int


@dataclass
class MapInfo:
    Name: str
    Duration: str
    Choice: int
    Team1: Team
    Team2: Team


@dataclass
class Round:
    team1_score: int
    team2_score: int
    first_attacker: str


@dataclass
class TableRow:
    RoundNumber: int
    Winner: str
    ResultType: str
    Team1Economy: str
    Team2Economy: str


@dataclass
class MapResult:
    winner: str
    first_attacker: str


def make_long_df(rounds: list[TableRow], first_attacker: str) -> list[Round]:
    data: list[Round] = []
    team1_score = 0
    team2_score = 0
    for row in rounds:
        if row.RoundNumber < 13:
            attacker = 'Team1' if first_attacker == 'Team1' else 'Team2'
            # first half
            if row.Winner == 't':
                team1_score += 1
                data.append(
                    Round(team1_score=team1_score, team2_score=team2_score, first_attacker=attacker)
                )
            else:
                team2_score += 1
                data.append(
                    Round(team1_score=team1_score, team2_score=team2_score, first_attacker=attacker)
                )
        else:
            # switch to second half
            attacker = 'Team2' if first_attacker == 'Team1' else 'Team1'
            if row.Winner == 'ct':
                team1_score += 1
                data.append(
                    Round(team1_score=team1_score, team2_score=team2_score, first_attacker=attacker)
                )
            else:
                team2_score += 1
                data.append(
                    Round(team1_score=team1_score, team2_score=team2_score, first_attacker=attacker)
                )
    return data
        

def identify_first_attacker(tsdata: pd.DataFrame, mapdata: MapInfo) -> str:
    first_half = tsdata.loc[:12, 'Winner'].value_counts().to_dict()
    second_half = tsdata.loc[12:, 'Winner'].value_counts().to_dict()
    team1_attack_first = first_half.get('t', 0) + second_half.get('ct', 0) == mapdata.Team1.score
    return 'Team1' if team1_attack_first else 'Team2'


def extract_map_data(
    tsdata: pd.DataFrame,
    mapdata: MapInfo,
) -> MapResult:
    winner = 'Team1' if mapdata.Team1.score > mapdata.Team2.score else 'Team2'
    first_attacker = identify_first_attacker(tsdata, mapdata)
    return MapResult(winner=winner, first_attacker=first_attacker)


def make_map_info(map_info: dict[str, str | int | dict[str, str | int]]) -> MapInfo:
    return MapInfo(
        Name=map_info['Name'],
        Duration=map_info['Duration'],
        Choice=map_info['Choice'],
        Team1=Team(
            name=map_info['Team1']['name'],
            score=map_info['Team1']['score'],
            attack=map_info['Team1']['attack'],
            defense=map_info['Team1']['defense'],
        ),
        Team2=Team(
            name=map_info['Team2']['name'],
            score=map_info['Team2']['score'],
            attack=map_info['Team2']['attack'],
            defense=map_info['Team2']['defense'],
        ),
    )


def add_columns(ldf: pd.DataFrame, mapinfo: MapInfo, mapresult: MapResult) -> pd.DataFrame:
    df = ldf.copy()
    df['team1_name'] = mapinfo.Team1.name
    df['team2_name'] = mapinfo.Team2.name
    df['map_name'] = mapinfo.Name
    df['winner'] = mapresult.winner

    df['encoded_team1_attack'] = df['first_attacker'].map({'Team1': 1, 'Team2': 0})
    df['encoded_team1_win'] = df['winner'].map({'Team1': 1, 'Team2': 0})
    return df


ENCODERS = {
    'encoded_team1_name': ('team1_name', LabelEncoder()),
    'encoded_team2_name': ('team2_name', LabelEncoder()),
    'encoded_map_name': ('map_name', LabelEncoder()),
}


def build_model(dff: pd.DataFrame) -> DecisionTreeClassifier:
    input_cols = ['team1_score', 'team2_score', 'encoded_team1_attack', 'encoded_team1_name', 'encoded_team2_name', 'encoded_map_name']
    target_col = 'encoded_team1_win'

    df = dff.copy()

    # make encoded columns
    for (new_col_name, (old_col_name, encoder)) in ENCODERS.items():
        df[new_col_name] = encoder.fit_transform(df[old_col_name])
    
    X_train, X_test, y_train, y_test = train_test_split(
        df[input_cols],
        df[target_col],
        test_size=0.25,
        random_state=42
    )

    model = DecisionTreeClassifier(max_depth=5)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f'Accuracy (score): {model.score(X_test, y_test)}')
    print(f'Accuracy (calc): {accuracy_score(y_test, y_pred)}')
    print(classification_report(y_test, y_pred))
    return model


def main():
    all_data: list[pd.DataFrame] = []
    for dir in Path("data").iterdir():
        if not dir.is_dir():
            continue
        for mapname in (dir / "maps").iterdir():
            ts_df = pd.read_csv(mapname / "time_series.csv")
            rounds: list[TableRow] = [TableRow(**row) for row in ts_df.to_dict(orient='records')]
            with open(mapname / "map_info.yaml") as f:
                raw_map_info: dict[str, str | int | dict[str, str | int]] = yaml.load(f, Loader=yaml.FullLoader)
                map_info = make_map_info(raw_map_info)
                map_data = extract_map_data(ts_df, map_info)
                long_data = make_long_df(rounds, map_data.first_attacker)
                long_df = pd.DataFrame(long_data)
                df = add_columns(long_df, map_info, map_data)
                all_data.append(df)
    
    all_df = pd.concat(all_data, ignore_index=True)
    all_df.to_csv("data/all_data.csv", index=False)
    print("Total records:", all_df.shape[0])

    model = build_model(all_df)
    print('Saving model...')
    with open('model.pkl', 'wb') as f:
        pickle.dump(model, f)

    print('Please input data to receive prediction:')
    team1_score = int(input('Team1 score: '))
    team2_score = int(input('Team2 score: '))
    attacker = input('Who is attacking? (Team1 or Team2): ')
    team1_name = input('Team1 name: ')
    team2_name = input('Team2 name: ')
    map_name = input('Map name: ')
    data = {
        'team1_score': team1_score,
        'team2_score': team2_score,
        'encoded_team1_attack': 1 if attacker == 'Team1' else 0,
        # this part could be better
        'encoded_team1_name': ENCODERS['encoded_team1_name'][1].transform([team1_name]),
        'encoded_team2_name': ENCODERS['encoded_team2_name'][1].transform([team2_name]),
        'encoded_map_name': ENCODERS['encoded_map_name'][1].transform([map_name]),
    }
    df = pd.DataFrame(data, index=[0])
    print(df)
    prediction = model.predict(df)
    print(f"Prediction: {'Team1' if prediction[0] == 1 else 'Team2'}")
    probabilities = model.predict_proba(df)
    # zero index because only one prediction
    team1_win_prob = probabilities[0][1]  # 1 index because team1 winning is a 1 result
    team2_win_prob = probabilities[0][0]  # 0 index because team2 winning is a 0 result
    print(f"Team1 win probability: {team1_win_prob}")
    print(f"Team2 win probability: {team2_win_prob}")


if __name__ == "__main__":
    main()
