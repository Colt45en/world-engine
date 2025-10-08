
            const result = (function() {
                
const message = "Hello from JavaScript!";
const numbers = [1, 2, 3, 4, 5];
const sum = numbers.reduce((a, b) => a + b, 0);
return { message, sum, numbers };

            })();
            
            console.log(JSON.stringify({
                status: 'success',
                result: result,
                timestamp: new Date().toISOString()
            }));
            