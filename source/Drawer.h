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
            std::cerr << "'DRAWING' DIRECTIVE IS ENABLED BUT THE SPECIFIED TEXT FONT (.ttf) WAS NOT FOUND. "
                << "MAKE SURE IT IS ALONGSIDE THE EXECUTABLE, OR IN THE ACTIVE DIRECTORY." << std::endl;
        }
    };

    void draw(Network* n, int step) {

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

        static std::vector<float> Xs(
            MAX_COMPLEX_CHILDREN_PER_COMPLEX + MAX_MEMORY_CHILDREN_PER_COMPLEX + MAX_SIMPLE_CHILDREN_PER_COMPLEX + 3
        );
        static std::vector<float> Ys(
            MAX_COMPLEX_CHILDREN_PER_COMPLEX + MAX_MEMORY_CHILDREN_PER_COMPLEX + MAX_SIMPLE_CHILDREN_PER_COMPLEX + 3
        );

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
        text.setFillColor(sf::Color::White);
        text.setString("NODE COLOR LEGEND:   Red = output   Green = input   Yellow = modulation    Blue = complex    White = simple     Pink = memory");
        text.setPosition(10.0f, w.getSize().y - 30.0f);
        w.draw(text);
        text.setString("Generation " + std::to_string(step));
        text.setPosition(w.getSize().x - 150.0f, 10.0f);
        w.draw(text);

        selfConnexion.setFillColor(sf::Color::Transparent);
        selfConnexion.setOutlineColor(sf::Color::White);
        selfConnexion.setOutlineThickness(1.0f);

        if (paused) {
            text.setString("PAUSED !  Press SPACE to resume.");
            text.setPosition(10.0f, 10.0f);
            w.draw(text);
        }


        float x0 = offset/2.0f, y0 = offset/1.5f;
        for (int i = (int)n->complexGenome.size(); i >= 0; i--) {
            ComplexNode_G* gNode = i == n->complexGenome.size() ? n->topNodeG.get() : n->complexGenome[i].get();
            if (gNode->phenotypicMultiplicity == 0) {continue;}
            if (x0 + offset > w.getSize().x) {
                x0 = offset; 
                y0 = 2.25f * offset;
            }
            
            text.setFillColor(sf::Color::White);
            if (i == n->complexGenome.size()) {
                text.setString("Top node, depth " + std::to_string(gNode->depth));
            }
            else {
                text.setString("Node n°" + std::to_string(gNode->position)  + ", depth " + std::to_string(gNode->depth));
            }
            text.setPosition(x0-1.25f*wheelRadius, y0 + 1.75f * wheelRadius);
            w.draw(text);
            text.setString("In size " + std::to_string(gNode->inputSize) + ", out size " + std::to_string(gNode->outputSize));
            text.setPosition(x0- 1.25f * wheelRadius, y0 + 2.25f * wheelRadius);
            w.draw(text);

            int _nNodes = gNode->complexChildren.size() + gNode->memoryChildren.size() + gNode->simpleChildren.size() + 3;
            float factor = 6.28f / (float) _nNodes;
            for (int j = 0; j < _nNodes; j++) {
                Xs[j] = x0 + wheelRadius * cosf(factor * (float)j);
                Ys[j] = y0 + wheelRadius * sinf(factor * (float)j);
            }

            // Indices in X and Y of the origin and destination nodes
            int _oID, _dID;
            NODE_TYPE oType, dType;
            for (int j = 0; j < gNode->internalConnexions.size(); j++) {
                oType = gNode->internalConnexions[j].originType;
                switch (oType) {
                case INPUT_NODE:
                    _oID = 0;
                    break;
                case OUTPUT:
                    _oID = 1;
                    break;
                case MODULATION:
                    _oID = 2;
                    break;
                case SIMPLE:
                    _oID = 3 + gNode->internalConnexions[j].originID;
                    break;
                case COMPLEX:
                    _oID = 3 + gNode->simpleChildren.size() + gNode->internalConnexions[j].originID;
                    break;
                case MEMORY:
                    _oID = 3 + gNode->simpleChildren.size() + gNode->complexChildren.size() + gNode->internalConnexions[j].originID;
                    break;
                }

                dType = gNode->internalConnexions[j].destinationType;
                switch (dType) {
                case INPUT_NODE:
                    _dID = 0;
                    break;
                case OUTPUT:
                    _dID = 1;
                    break;
                case MODULATION:
                    _dID = 2;
                    break;
                case SIMPLE:
                    _dID = 3 + gNode->internalConnexions[j].destinationID;
                    break;
                case COMPLEX:
                    _dID = 3 + gNode->simpleChildren.size() + gNode->internalConnexions[j].destinationID;
                    break;
                case MEMORY:
                    _dID = 3 + gNode->simpleChildren.size() + gNode->complexChildren.size() + gNode->internalConnexions[j].destinationID;
                    break;
                }

                if (_oID != _dID) {
                    line[0].position.x = Xs[_oID] + nodeRadius; // + node radius to be at the node's center.
                    line[0].position.y = Ys[_oID] + nodeRadius;
                    line[1].position.x = Xs[_dID] + nodeRadius;
                    line[1].position.y = Ys[_dID] + nodeRadius;

                    w.draw(line, 2, sf::Lines);
                }
                else {
                    float x = Xs[_oID] + nodeRadius * (Xs[_oID] - x0) / wheelRadius;
                    float y = Ys[_oID] + nodeRadius * (Ys[_oID] - y0) / wheelRadius;
                    selfConnexion.setPosition(x,y);
                    w.draw(selfConnexion);
                }
            }


            node.setFillColor(sf::Color::Red); //output
            node.setPosition(Xs[1], Ys[1]);
            w.draw(node);

            node.setFillColor(sf::Color::Green); //input
            node.setPosition(Xs[0], Ys[0]);
            w.draw(node);

            node.setFillColor(sf::Color::Yellow); //modulation
            node.setPosition(Xs[2], Ys[2]);
            w.draw(node);

            node.setFillColor(sf::Color::White);
            text.setFillColor(sf::Color::Black);
            for (int j = 0; j < gNode->simpleChildren.size(); j++) {
                int id = 3 + j;
                node.setPosition(Xs[id], Ys[id]);
    
                w.draw(node);

                text.setString(std::to_string(gNode->simpleChildren[j]->position));
                text.setPosition(Xs[id] + nodeRadius / 4.0f, Ys[id] + nodeRadius / 4.0f);
                w.draw(text);
            }

            node.setFillColor(sf::Color::Cyan);
            for (int j = 0; j < gNode->complexChildren.size(); j++) {
                int id = 3 + gNode->simpleChildren.size() + j;
                node.setPosition(Xs[id], Ys[id]);

                w.draw(node);

                text.setString(std::to_string(gNode->complexChildren[j]->position));
                text.setPosition(Xs[id] + nodeRadius / 4.0f, Ys[id] + nodeRadius / 4.0f);
                w.draw(text);
            }

            node.setFillColor(sf::Color::Magenta);
            for (int j = 0; j < gNode->memoryChildren.size(); j++) {
                int id = 3 + gNode->simpleChildren.size() + gNode->complexChildren.size() + j;
                node.setPosition(Xs[id], Ys[id]);

                w.draw(node);

                text.setString(std::to_string(gNode->memoryChildren[j]->position));
                text.setPosition(Xs[id] + nodeRadius / 4.0f, Ys[id] + nodeRadius / 4.0f);
                w.draw(text);
            }

            x0 += offset*1.25f;
        }

        w.display();
    }
};