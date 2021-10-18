from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Dict

from pyts.image import GramianAngularField
from pyts.preprocessing import StandardScaler, InterpolationImputer

ROOT = Path(__file__).parent.parent.absolute()
DATA_DIR = ROOT / "data" / "orig"


@dataclass
class dataset:
    _class: str
    data: field(default_factory=list)


@dataclass
class dataset_image:
    _class: str
    data: GramianAngularField


def data_loader(filename: Path) -> List[Dict]:
    contents = filename.read_text()
    datasets: List[dataclass] = []

    for line in contents.splitlines():
        line_words = line.split()
        if len(line_words) == 1:
            datasets.append(dataset(_class=line_words[0], data=list()))
        elif len(line_words) > 0:
            datasets[-1].data.append([int(word) for word in line_words])

    return datasets


def preprocess(dataset: List[List]) -> List[List]:
    scaler = StandardScaler()
    return scaler.transform(dataset)


def main():
    file = DATA_DIR / "lp1.data"
    datas = data_loader(file)
    i_datas = []

    transformer = GramianAngularField()
    for datum in datas:
        t_img = transformer.transform(preprocess(datum.data)).flatten()
        i_datas.append(dataset_image(_class=datum._class, data=t_img))

    print(i_datas)


if __name__ == "__main__":
    main()
