#pragma once

#include "Population.h"
#include <SFML/Graphics.hpp>
#include <string>
#include <iostream>

class Drawer {
private :
    sf::RenderWindow w;
    sf::Font font;
public:
    bool paused;
    Drawer(int w, int h) :
        w(sf::VideoMode(w, h), "Fittest Genotype")
    {
        paused = false;
        if (!font.loadFromFile("Roboto-Black.ttf"))
        {
            std::cerr << "DRAWING OPTION IS ENABLED BUT THE SPECIFIED TEXT FONT (.ttf) WAS NOT FOUND."
                << "MAKE SURE IT IS ALONGSIDE THE EXECUTABLE, OR IN THE ACTIVE DIRECTORY." << std::endl;
        }
    };

    void draw(Network* n) {

        constexpr float nodeRadius = 10.0f;
        constexpr float wheelRadius = 40.0f;
        constexpr float offset = 130.0f;

        static sf::CircleShape node(10.0f);
        static sf::CircleShape selfConnexion(nodeRadius);
        static sf::Vertex line[] =
        {
            sf::Vertex(sf::Vector2f(0.0f, 0.0f)),
            sf::Vertex(sf::Vector2f(0.0f, 0.0f))
        };
        static sf::Text text;

        static std::vector<float> Xs(MAX_CHILDREN_PER_BLOCK + 3);
        static std::vector<float> Ys(MAX_CHILDREN_PER_BLOCK + 3);

        w.clear();
        sf::Event event;
        while (w.pollEvent(event))
        {
            if (event.type == sf::Event::Closed)
                w.close();
            else if (event.type == sf::Event::KeyReleased && event.key.code == sf::Keyboard::Space) {
                paused = !paused;
            }
        }


        text.setFont(font);
        text.setCharacterSize(15);

        selfConnexion.setFillColor(sf::Color::Transparent);
        selfConnexion.setOutlineColor(sf::Color::White);
        selfConnexion.setOutlineThickness(1.0f);

        if (paused) {
            text.setFillColor(sf::Color::White);
            text.setString("PAUSED !  Press SPACE to resume.");
            text.setPosition(10.0f, 10.0f);
            w.draw(text);
        }


        float x0 = offset/2.0f, y0 = offset/1.5f;
        for (int i = (int)n->genome.size(); i >= n->nSimpleNeurons; i--) {
            if (x0 + offset > w.getSize().x) {
                x0 = offset; 
                y0 = 2.75f * offset;
            }
            GenotypeNode* gNode;
            text.setFillColor(sf::Color::White);
            if (i == n->genome.size()) {
                gNode = n->topNodeG.get();
                text.setString("Top node, depth " + std::to_string(gNode->depth));
            }
            else {
                gNode = n->genome[i].get();
                text.setString("Node n°" + std::to_string(gNode->position)  + ", depth " + std::to_string(gNode->depth));
            }
            text.setPosition(x0-1.25f*wheelRadius, y0 + 1.75f * wheelRadius);
            w.draw(text);
            text.setString("In size " + std::to_string(gNode->inputSize) + ", out size " + std::to_string(gNode->outputSize));
            text.setPosition(x0- 1.25f * wheelRadius, y0 + 2.25f * wheelRadius);
            w.draw(text);
            text.setFillColor(sf::Color::Black);


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

                if (oID != dID) {
                    line[0].position.x = Xs[oID] + nodeRadius; // + node radius to be at the node's center.
                    line[0].position.y = Ys[oID] + nodeRadius;
                    line[1].position.x = Xs[dID] + nodeRadius;
                    line[1].position.y = Ys[dID] + nodeRadius;

                    w.draw(line, 2, sf::Lines);
                }
                else {
                    float x = Xs[oID] + nodeRadius * (Xs[oID] - x0) / wheelRadius;
                    float y = Ys[oID] + nodeRadius * (Ys[oID] - y0) / wheelRadius;
                    selfConnexion.setPosition(x,y);
                    w.draw(selfConnexion);
                }
            }

            for (int j = 0; j < gNode->children.size(); j++) {
                node.setPosition(Xs[j], Ys[j]);
                if (gNode->children[j]->isSimpleNeuron) {
                    node.setFillColor(sf::Color::White);
                }
                else {
                    node.setFillColor(sf::Color::Cyan);
                }
                w.draw(node);

                text.setString(std::to_string(gNode->children[j]->position));
                text.setPosition(Xs[j] + nodeRadius / 4.0f, Ys[j] + nodeRadius / 4.0f);
                w.draw(text);
            }

            node.setFillColor(sf::Color::Red); //output
            node.setPosition(Xs[gNode->children.size()], Ys[gNode->children.size()]);
            w.draw(node);

            node.setFillColor(sf::Color::Green); //input
            node.setPosition(Xs[gNode->children.size() + 1], Ys[gNode->children.size() + 1]);
            w.draw(node);

            node.setFillColor(sf::Color::Yellow); //modulation
            node.setPosition(Xs[gNode->children.size() + 2], Ys[gNode->children.size() + 2]);
            w.draw(node);

            x0 += offset*1.25f;
        }

        w.display();
    }
};