import random


def gen_problem_10(rng: random.Random) -> str:
    n = rng.randint(1, 10)
    values = [str(rng.randint(-20, 20)) for _ in range(n)]
    return f"{n}\n" + " ".join(values) + "\n"


def gen_problem_16(rng: random.Random) -> str:
    b = rng.randint(0, 99999)
    return f"{b}\n"


def gen_problem_43(rng: random.Random) -> str:
    m = rng.randrange(4, 200, 2)
    return f"{m}\n"


def gen_problem_66(rng: random.Random) -> str:
    y = rng.randint(1900, 2099)
    m = rng.randint(1, 12)
    d = rng.randint(1, 28)
    return f"{y} {m} {d}\n"


PROBLEM_GENERATORS = {
    10: gen_problem_10,
    16: gen_problem_16,
    43: gen_problem_43,
    66: gen_problem_66,
}


def get_input_generator(problem_id: int):
    if problem_id not in PROBLEM_GENERATORS:
        raise NotImplementedError(f"No input generator for problem {problem_id}")
    return PROBLEM_GENERATORS[problem_id]
