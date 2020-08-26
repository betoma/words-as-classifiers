import random
from collections import Counter
import numpy as np
from scipy.stats import beta
from tqdm import tqdm
from num2words import num2words


class FakeEntity:
    possible_types = ["cat", "dog", "man", "woman", "stool", "chair"]
    possible_colors = ["white", "black", "brown"]

    def __init__(self, obj_type, color=None):
        """
        a cat, dog, man, woman, tree, stool, or chair with a random color
        """
        if obj_type not in self.possible_types:
            raise ValueError
        if color is not None and color not in self.possible_colors:
            raise ValueError
        self.my_type = obj_type
        self.my_color = color

    def __repr__(self):
        return f"Ent({self.my_type}, {self.my_color})"

    def __eq__(self, other):
        return self.my_type == other.my_type and self.my_color == other.my_color

    def __hash__(self):
        return hash((self.my_type, self.my_color))

    @classmethod
    def random(cls):
        return cls(
            random.choice(cls.possible_types), random.choice(cls.possible_colors)
        )

    @classmethod
    def random_colorless(cls):
        return cls(random.choice(cls.possible_types))


class FakePhoto:
    def __init__(self):
        self.contents = Counter()
        self.cont_list = []

    def __repr__(self):
        return "Photo{}".format(str(dict(self.contents)))

    def __str__(self):
        return "Photo{}".format(str(dict(self.contents)))

    def __contains__(self, other):
        check = []
        for x in other.contents:
            if x.my_color is None:
                noun_sum = 0
                for c in self.contents:
                    if c.my_type == x.my_type:
                        noun_sum += self.contents[c]
                if noun_sum >= other.contents[x]:
                    check.append(True)
                else:
                    check.append(False)
            else:
                if self.contents[x] >= other.contents[x]:
                    check.append(True)
                else:
                    check.append(False)
        return all(check)

    @classmethod
    def random(cls):
        pic = cls()
        n = random.randint(1, 25)
        for _ in range(n):
            pic.add(FakeEntity.random())
        return pic

    def add(self, entity: FakeEntity):
        self.contents[entity] += 1
        self.cont_list.append(entity)


class FakeData:
    n_det = {k: ("a" if k == 1 else num2words(k)) for k in range(1, 10)}
    n_verb = {k: ("is" if k == 1 else "are") for k in range(1, 10)}
    n_form = {
        "cat": {k: ("cat" if k == 1 else "cats") for k in range(1, 10)},
        "dog": {k: ("dog" if k == 1 else "dogs") for k in range(1, 10)},
        "man": {k: ("man" if k == 1 else "men") for k in range(1, 10)},
        "woman": {k: ("woman" if k == 1 else "women") for k in range(1, 10)},
        "stool": {k: ("stool" if k == 1 else "stools") for k in range(1, 10)},
        "chair": {k: ("chair" if k == 1 else "chairs") for k in range(1, 10)},
    }
    type_correspondences = {
        "cat": {
            "very close": {"dog"},
            "close-ish": {"man", "woman"},
            "not close": {"stool", "chair"},
        },
        "dog": {
            "very close": {"cat"},
            "close-ish": {"man", "woman"},
            "not close": {"stool", "chair"},
        },
        "man": {
            "very close": {"woman"},
            "close-ish": {"cat", "dog"},
            "not close": {"stool", "chair"},
        },
        "woman": {
            "very close": {"man"},
            "close-ish": {"cat", "dog"},
            "not close": {"stool", "chair"},
        },
        "stool": {
            "very close": {"chair"},
            "close-ish": {},
            "not close": {"man", "woman", "dog", "cat"},
        },
        "chair": {
            "very close": {"stool"},
            "close-ish": {},
            "not close": {"man", "woman", "dog", "cat"},
        },
        "black": {"close": {"brown"}, "not": {"white"}},
        "white": {"close": {}, "not": {"black", "brown"}},
        "brown": {"close": {"black"}, "not": {"white"}},
    }
    exact_dist = beta(5, 1.3)
    close_dist = beta(5, 1.6)
    ish_dist = beta(5, 2.5)
    not_dist = beta(3.5, 5)
    class_cols = FakeEntity.possible_types + FakeEntity.possible_colors

    def __init__(self, n: int = 10):
        self.photos = []
        for _ in range(n):
            self.photos.append(FakePhoto.random())
        self.checks = []
        self.utterances = []
        self.correct = []
        self.class_output = []
        for p in tqdm(self.photos):
            cl_ent_list = []
            for ent in p.cont_list:
                ent_val_list = []
                for t in self.class_cols:
                    if t in FakeEntity.possible_types:
                        if t == ent.my_type:
                            val = self.exact_dist.rvs()
                        elif t in self.type_correspondences[ent.my_type]["very close"]:
                            val = self.close_dist.rvs()
                        elif t in self.type_correspondences[ent.my_type]["close-ish"]:
                            val = self.ish_dist.rvs()
                        elif t in self.type_correspondences[ent.my_type]["not close"]:
                            val = self.not_dist.rvs()
                    elif t in FakeEntity.possible_colors:
                        if t == ent.my_color:
                            val = self.exact_dist.rvs()
                        elif t in self.type_correspondences[ent.my_color]["close"]:
                            val = self.close_dist.rvs()
                        elif t in self.type_correspondences[ent.my_color]["not"]:
                            val = self.not_dist.rvs()
                    ent_val_list.append(val)
                cl_ent_list.append(ent_val_list)
            self.class_output.append(np.array(cl_ent_list))
            if random.randint(0, 2) > 1:
                if random.randint(0, 1):
                    rand_ent = random.choice(p.cont_list)
                    rand_n = random.randint(1, p.contents[rand_ent])
                else:
                    bla = random.choice(p.cont_list)
                    rand_ent = FakeEntity(bla.my_type)
                    rand_n = random.randint(1, p.contents[bla])
            else:
                if random.randint(0, 1):
                    rand_ent = FakeEntity.random_colorless()
                else:
                    rand_ent = FakeEntity.random()
                rand_n = random.randint(1, 6)
            pic = FakePhoto()
            for _ in range(rand_n):
                pic.add(rand_ent)
            self.checks.append(pic)
            if pic in p:
                self.correct.append(True)
            else:
                self.correct.append(False)
            if rand_ent.my_color:
                color = f"{rand_ent.my_color} "
            else:
                color = ""
            self.utterances.append(
                "There {} {} {}{}.".format(
                    self.n_verb[rand_n],
                    self.n_det[rand_n],
                    color,
                    self.n_form[rand_ent.my_type][rand_n],
                )
            )

    def __str__(self):
        s = ""
        for i, p in enumerate(self.photos):
            s += f"{p}\t{self.checks[i]}\t{self.utterances[i]}\t{self.correct[i]}\t{self.class_output}\n"
        return s


if __name__ == "__main__":
    d = FakeData(1)
    print(d)
