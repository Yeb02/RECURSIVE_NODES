#pragma once

#include "Population.h"
#include <SFML/Graphics.hpp>

class Drawer {
    sf::RenderWindow w;
public:
    Drawer(int w, int h) :
        w(sf::VideoMode(w, h), "Top Node")
    {};

    void draw(Network* n) {
        w.clear();
        sf::Event event;
        while (w.pollEvent(event))
        {
            if (event.type == sf::Event::Closed)
                w.close();
        }

        static sf::CircleShape node(10.0f);
        static sf::Vertex line[] =
        {
            sf::Vertex(sf::Vector2f(0.0f, 0.0f)),
            sf::Vertex(sf::Vector2f(0.0f, 0.0f))
        };

        constexpr float wheelRadius = 40.0f;
        constexpr float offset = 130.0f;

        static std::vector<float> Xs(MAX_CHILDREN_PER_BLOCK + 3);
        static std::vector<float> Ys(MAX_CHILDREN_PER_BLOCK + 3);


        float x0 = offset, y0 = offset;
        for (int i = (int)n->genome.size(); i >= n->nSimpleNeurons; i--) {
            GenotypeNode* gNode = i == n->genome.size() ? n->topNodeG.get() : n->genome[i].get();
            node.setFillColor(sf::Color::Blue);

            float factor = 6.28f / ((float)gNode->children.size() + 3.0f);
            for (int j = 0; j < gNode->children.size() + 3; j++) {
                Xs[j] = x0 + wheelRadius * cosf(factor * (float)j);
                Ys[j] = y0 + wheelRadius * sinf(factor * (float)j);
            }

            int oID, dID;
            for (int j = 0; j < gNode->childrenConnexions.size(); j++) {
                oID = gNode->childrenConnexions[j].originID;
                oID = oID == INPUT_ID ? (int)gNode->children.size() + 1 : oID;
                dID = gNode->childrenConnexions[j].destinationID;
                dID = dID == MODULATION_ID ? (int)gNode->children.size() + 2 : dID;

                line[0].position.x = Xs[oID] + 10.0f; // + circle radius
                line[0].position.y = Ys[oID] + 10.0f;
                line[1].position.x = Xs[dID] + 10.0f;
                line[1].position.y = Ys[dID] + 10.0f;

                w.draw(line, 2, sf::Lines);
            }

            for (int j = 0; j < gNode->children.size(); j++) {
                node.setPosition(sf::Vector2f(Xs[j], Ys[j]));
                if (gNode->children[j]->isSimpleNeuron) {
                    node.setFillColor(sf::Color::White);
                }
                else {
                    node.setFillColor(sf::Color::Blue);
                }
                w.draw(node);
            }

            node.setFillColor(sf::Color::Red); //output
            node.setPosition(sf::Vector2f(Xs[gNode->children.size()], Ys[gNode->children.size()]));
            w.draw(node);

            node.setFillColor(sf::Color::Green); //input
            node.setPosition(sf::Vector2f(Xs[gNode->children.size() + 1], Ys[gNode->children.size() + 1]));
            w.draw(node);

            node.setFillColor(sf::Color::Yellow); //modulation
            node.setPosition(sf::Vector2f(Xs[gNode->children.size() + 2], Ys[gNode->children.size() + 2]));
            w.draw(node);

            x0 += offset;
        }

        w.display();
    }
};