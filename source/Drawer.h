#pragma once

#include "Population.h"
#include <SFML/Graphics.hpp>

class Drawer {
    sf::RenderWindow& w;
public:
    Drawer(sf::RenderWindow& w) : w(w) {};
    void draw(Network* n) {
        static sf::CircleShape node(10.0f);
        static sf::Vertex line[] =
        {
            sf::Vertex(sf::Vector2f(0.0f, 0.0f)),
            sf::Vertex(sf::Vector2f(0.0f, 0.0f))
        };

        constexpr float wheelRadius = 40.0f;
        constexpr float offset = 130.0f;

        static std::vector<float> Xs(MAX_CHILDREN_PER_BLOCK + 2);
        static std::vector<float> Ys(MAX_CHILDREN_PER_BLOCK + 2);


        float x0 = offset, y0 = offset;
        for (int i = (int)n->genome.size() - 1; i >= n->nSimpleNeurons; i--) {

            node.setFillColor(sf::Color::Blue);

            float factor = 6.28f / ((float)n->genome[i]->children.size() + 2.0f);
            for (int j = 0; j < n->genome[i]->children.size() + 2; j++) {
                Xs[j] = x0 + wheelRadius * cosf(factor * (float)j);
                Ys[j] = y0 + wheelRadius * sinf(factor * (float)j);
            }

            int oID, dID;
            for (int j = 0; j < n->genome[i]->childrenConnexions.size(); j++) {
                oID = n->genome[i]->childrenConnexions[j].originID;
                oID = oID == -1 ? (int)n->genome[i]->children.size() + 1 : oID;
                dID = n->genome[i]->childrenConnexions[j].destinationID;

                line[0].position.x = Xs[oID] + 10.0f; // + circle radius
                line[0].position.y = Ys[oID] + 10.0f;
                line[1].position.x = Xs[dID] + 10.0f;
                line[1].position.y = Ys[dID] + 10.0f;

                w.draw(line, 2, sf::Lines);
            }

            for (int j = 0; j < n->genome[i]->children.size(); j++) {
                node.setPosition(sf::Vector2f(Xs[j], Ys[j]));
                if (n->genome[i]->children[j]->isSimpleNeuron) {
                    node.setFillColor(sf::Color::White);
                }
                else {
                    node.setFillColor(sf::Color::Blue);
                }
                w.draw(node);
            }

            node.setFillColor(sf::Color::Red); //output
            node.setPosition(sf::Vector2f(Xs[n->genome[i]->children.size()], Ys[n->genome[i]->children.size()]));
            w.draw(node);

            node.setFillColor(sf::Color::Green); //input
            node.setPosition(sf::Vector2f(Xs[n->genome[i]->children.size() + 1], Ys[n->genome[i]->children.size() + 1]));
            w.draw(node);

            x0 += offset;
        }
    }
};