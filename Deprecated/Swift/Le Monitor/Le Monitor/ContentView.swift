//
//  ContentView.swift
//  Le Monitor
//
//  Created by Matthew Ziyu Su on 9/20/24.
//

import SwiftUI

struct Message: Identifiable {
    let id = UUID()
    let content: String
    let isUser: Bool
}

struct ContentView: View {
    @State private var messages: [Message] = []
    @State private var newMessage: String = ""
    
    var body: some View {
        VStack {
            // Chat messages
            ScrollView {
                LazyVStack(alignment: .leading, spacing: 10) {
                    ForEach(messages) { message in
                        MessageBubble(message: message)
                    }
                }
                .padding()
            }
            
            // Input area
            HStack {
                TextField("Type a message...", text: $newMessage)
                    .textFieldStyle(RoundedBorderTextFieldStyle())
                    .padding(.horizontal)
                
                Button(action: sendMessage) {
                    Image(systemName: "paperplane.fill")
                }
                .padding(.trailing)
            }
            .padding(.bottom)
        }
    }
    
    func sendMessage() {
        guard !newMessage.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { return }
        let userMessage = Message(content: newMessage, isUser: true)
        messages.append(userMessage)
        
        // Simulate AI response (you'd replace this with actual AI logic)
        DispatchQueue.main.asyncAfter(deadline: .now() + 1) {
            let aiMessage = Message(content: "This is an AI response.", isUser: false)
            messages.append(aiMessage)
        }
        
        newMessage = ""
    }
}

struct MessageBubble: View {
    let message: Message
    
    var body: some View {
        HStack {
            if message.isUser { Spacer() }
            Text(message.content)
                .padding(10)
                .background(message.isUser ? Color.blue : Color.gray)
                .foregroundColor(.white)
                .cornerRadius(10)
            if !message.isUser { Spacer() }
        }
    }
}

#Preview {
    ContentView()
}
