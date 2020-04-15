from sklearn.model_selection import ParameterGrid

from tagger.model_solver import Solver


def grid_search():
    """Method helps to make a grid search"""
    grid = {
        'E_DIM': [64],
        'H_DIM': [64],
        'lr': [0.1, 0.01],
        'token_mode': ['form', 'lemma'],
        'epochs': [50],
    }

    # datapath = 'Taiga'
    datapath = 'ENG_EWT'

    for conf, hypers in enumerate(sorted(ParameterGrid(grid), key=lambda x: list(x.values()))):
        solver = Solver(conf, hypers, datapath)
        solver.train()
        results = solver.evaluate()
        print('Evaluation results:')
        print("\n".join(results))


def train():
    hypers = {
        'E_DIM': 64,
        'H_DIM': 64,
        'lr': 0.1,
        'token_mode': 'lemma',
        'epochs': 50,
    }

    datapath = 'Taiga'

    solver = Solver(4,hypers, datapath)
    solver.train()
    results = solver.evaluate()
    print('Evaluation results:')
    print("\n".join(results))


if __name__ == '__main__':
    # grid_search()
    train()