# 🔐 Secure Multi-Room Chat System

A real-time communication platform built using Python socket programming with SSL/TLS encryption, file transfer, and performance monitoring.

## 📌 Project Overview
This project implements a multi-room chat system that allows users to communicate in real-time, create/join rooms, transfer files securely, and monitor performance.

## 🚀 Features
- SSL/TLS Encryption
- Multi-room chat support
- File transfer (up to 10MB)
- Performance monitoring (latency, throughput)
- Tkinter GUI client
- Multi-threaded server

## 🛠️ Technologies Used
- Python 3
- Socket Programming
- SSL/TLS
- Threading
- Tkinter
- JSON
- Base64
- Queue

## 🏗️ Architecture
Three-tier architecture:
- Client (GUI + Socket + SSL)
- Network (TCP with TLS)
- Server (Dispatcher + Rooms + Client Manager)

## ▶️ How to Run
1. Start server:
   python server.py

2. Start client:
   python client_gui.py

3. Connect using localhost:12345

## 👩‍💻 Authors
Deepthi V (PES2UG24CS150)
Dhanyashree K M (PES2UG24CS156)
