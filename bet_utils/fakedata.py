import random
from collections import Counter

import numpy as np
from num2words import num2words
from scipy.stats import beta

from .boxent import BoxofEntities


class FakeWorld(BoxofEntities):
    def __init__(
        self,
        n_confusers: """int between 1 and 10""" = 2,
        n_colors: """int between 1 and 10""" = 3,
        **kwargs,
    ):
        super(FakeWorld, self).__init__(**kwargs)
        if n_confusers <= 0 or n_confusers > 10 or n_colors <= 0 or n_colors > 10:
            raise ValueError("Value for number of confusers is ")
        self.possible_types = list(self.all_types)[0 : n_confusers * 3]
        self.possible_colors = list(self.all_colors)[0:n_colors]
        self.vocab_size = len(self.possible_types) + len(self.possible_colors)


class FakeEntity:
    def __init__(self, world: FakeWorld, obj_type, color=None):
        """
        an entity with a given color
        """
        if obj_type not in world.possible_types:
            raise ValueError
        if color is not None and color not in world.possible_colors:
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
    def random(cls, world: FakeWorld):
        return cls(
            world,
            random.choice(world.possible_types),
            random.choice(world.possible_colors),
        )

    @classmethod
    def random_colorless(cls, world: FakeWorld):
        return cls(world, random.choice(world.possible_types))


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
    def random(cls, world: FakeWorld):
        pic = cls()
        n = random.randint(1, 25)
        for _ in range(n):
            pic.add(FakeEntity.random(world))
        return pic

    def add(self, entity: FakeEntity):
        self.contents[entity] += 1
        self.cont_list.append(entity)


class FakeData:
    n_det = {k: ("A/AN" if k == 1 else num2words(k)) for k in range(1, 20)}
    n_verb = {k: ("is" if k == 1 else "are") for k in range(1, 20)}

    def __init__(self, world: FakeWorld, n: int = 10):
        self.class_cols = world.possible_types + world.possible_colors
        self.arr_cols = {k: i for i, k in enumerate(self.class_cols)}
        self.n_form = {
            word: {
                k: word
                if k == 1
                else world.irregular_plurals[word]
                if word in world.irregular_plurals
                else word + "s"
                for k in range(1, 20)
            }
            for word in world.possible_types
        }
        self.photos = []
        for _ in range(n):
            self.photos.append(FakePhoto.random(world))
        self.checks = []
        self.utterances = []
        self.correct = []
        self.class_output = []
        for p in self.photos:
            cl_ent_list = []
            for ent in p.cont_list:
                ent_val_list = []
                for t in self.class_cols:
                    if t in world.possible_types:
                        val = world.all_types[ent.my_type][t].rvs()
                    elif t in world.possible_colors:
                        val = world.all_colors[ent.my_color][t].rvs()
                    ent_val_list.append(val)
                cl_ent_list.append(ent_val_list)
            self.class_output.append(np.array(cl_ent_list))
            if random.randint(0, 2) > 1:
                if random.randint(0, 1):
                    rand_ent = random.choice(p.cont_list)
                    rand_n = random.randint(1, p.contents[rand_ent])
                else:
                    bla = random.choice(p.cont_list)
                    rand_ent = FakeEntity(world, bla.my_type)
                    rand_n = random.randint(1, p.contents[bla])
            else:
                if random.randint(0, 1):
                    rand_ent = FakeEntity.random_colorless(world)
                else:
                    rand_ent = FakeEntity.random(world)
                rand_n = random.randint(1, 6)
            pic = FakePhoto()
            for _ in range(rand_n):
                pic.add(rand_ent)
            self.checks.append(pic)
            if pic in p:
                self.correct.append(1)
            else:
                self.correct.append(0)
            if rand_ent.my_color:
                color = f"{rand_ent.my_color} "
                first_word = color
            else:
                color = ""
                first_word = rand_ent.my_type
            if rand_n == 1:
                if first_word[0] in ["a", "e", "i", "o", "u"]:
                    determiner = "an"
                else:
                    determiner = "a"
            else:
                determiner = self.n_det[rand_n]
            self.utterances.append(
                "There {} {} {}{}.".format(
                    self.n_verb[rand_n],
                    determiner,
                    color,
                    self.n_form[rand_ent.my_type][rand_n],
                )
            )

    def __str__(self):
        s = ""
        for i, p in enumerate(self.photos):
            s += f"{p}\t{self.checks[i]}\t{self.utterances[i]}\t{self.correct[i]}\t{self.class_output}\n"
        return s


# if __name__ == "__main__":
#     w = FakeWorld()
#     d = FakeData(w, 1)
#     print(d)
