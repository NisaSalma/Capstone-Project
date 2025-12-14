using System;
using System.Net.Http;
using System.Text;
using System.Text.Json;
using System.Threading.Tasks;

class Program
{
    static async Task Main(string[] args)
    {
        Console.WriteLine("Chatbot Tambang (Llama3 via Ollama)");
        Console.WriteLine("-----------------------------------");

        using var client = new HttpClient();

        while (true)
        {
            Console.Write("\nAnda: ");
            var userInput = Console.ReadLine();

            if (string.IsNullOrWhiteSpace(userInput)) continue;
            if (userInput.ToLower() == "exit") break;

            var body = new
            {
                model = "llama3.2",
                prompt = userInput,
                stream = false
            };

            var json = JsonSerializer.Serialize(body);
            var content = new StringContent(json, Encoding.UTF8, "application/json");

            var response = await client.PostAsync("http://localhost:11434/api/generate", content);
            var responseString = await response.Content.ReadAsStringAsync();

            Console.WriteLine("\nChatbot:");
            Console.WriteLine(responseString);
        }
    }
}
