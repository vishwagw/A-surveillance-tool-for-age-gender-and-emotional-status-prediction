import { useState, useEffect } from 'react';
import { Smile, Camera, RefreshCw, UserCheck } from 'lucide-react';

export default function FacialRecognitionSimulator() {
  const [analyzing, setAnalyzing] = useState(false);
  const [results, setResults] = useState(null);
  const [cameraActive, setCameraActive] = useState(false);
  
  // Simulate facial recognition analysis
  const analyzeImage = () => {
    setAnalyzing(true);
    
    // Simulate processing time
    setTimeout(() => {
      // Generate realistic but random results
      const simulatedResults = {
        age: Math.floor(Math.random() * 60) + 18, // 18-78
        gender: Math.random() > 0.5 ? 'Male' : 'Female',
        emotionalQuotient: Math.floor(Math.random() * 50) + 75, // 75-125
        emotions: {
          happiness: Math.floor(Math.random() * 100),
          sadness: Math.floor(Math.random() * 40),
          surprise: Math.floor(Math.random() * 60),
          neutral: Math.floor(Math.random() * 80)
        }
      };
      
      setResults(simulatedResults);
      setAnalyzing(false);
    }, 2000);
  };
  
  const startCamera = () => {
    setCameraActive(true);
    // In a real application, this would activate the camera
    setTimeout(analyzeImage, 1500);
  };
  
  const resetDemo = () => {
    setCameraActive(false);
    setResults(null);
  };
  
  return (
    <div className="bg-gray-100 p-6 rounded-lg shadow-lg max-w-md mx-auto">
      <h2 className="text-2xl font-bold mb-4 text-center text-blue-600">Facial Recognition System</h2>
      <p className="text-sm text-gray-500 mb-6 text-center">Demonstrating age, gender, and emotional intelligence detection</p>
      
      <div className="bg-white rounded-lg overflow-hidden shadow mb-6">
        <div className="relative aspect-video bg-gray-200 flex items-center justify-center">
          {!cameraActive ? (
            <Camera size={48} className="text-gray-400" />
          ) : (
            <div className="w-full h-full">
              <img 
                src="/api/placeholder/400/300" 
                alt="Camera feed placeholder" 
                className="w-full h-full object-cover"
              />
              {analyzing && (
                <div className="absolute inset-0 bg-black bg-opacity-50 flex items-center justify-center">
                  <div className="text-white text-center">
                    <RefreshCw size={40} className="animate-spin mx-auto mb-2" />
                    <p>Analyzing facial features...</p>
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
        
        {results && (
          <div className="p-4 border-t">
            <h3 className="font-semibold text-center mb-2 flex items-center justify-center">
              <UserCheck className="mr-2 text-green-500" />
              Analysis Results
            </h3>
            
            <div className="grid grid-cols-2 gap-4">
              <div className="bg-blue-50 p-3 rounded">
                <p className="text-sm text-gray-600">Age</p>
                <p className="text-xl font-bold">{results.age} years</p>
              </div>
              
              <div className="bg-purple-50 p-3 rounded">
                <p className="text-sm text-gray-600">Gender</p>
                <p className="text-xl font-bold">{results.gender}</p>
              </div>
              
              <div className="bg-green-50 p-3 rounded col-span-2">
                <p className="text-sm text-gray-600">Emotional Intelligence (EQ)</p>
                <p className="text-xl font-bold">{results.emotionalQuotient}</p>
                <div className="mt-2 h-3 bg-gray-200 rounded-full overflow-hidden">
                  <div 
                    className="h-full bg-green-500 rounded-full" 
                    style={{ width: `${Math.min(100, results.emotionalQuotient/1.5)}%` }}
                  ></div>
                </div>
                <p className="text-xs text-gray-500 mt-1">
                  {results.emotionalQuotient < 90 ? 'Developing' : 
                   results.emotionalQuotient < 110 ? 'Average' : 'Advanced'}
                </p>
              </div>
            </div>
            
            <div className="mt-4">
              <h4 className="text-sm font-medium mb-2">Detected Emotions</h4>
              <div className="space-y-2">
                {Object.entries(results.emotions).map(([emotion, value]) => (
                  <div key={emotion} className="flex items-center">
                    <span className="text-sm capitalize w-24">{emotion}</span>
                    <div className="flex-1 h-2 bg-gray-200 rounded-full overflow-hidden">
                      <div 
                        className={`h-full rounded-full ${
                          emotion === 'happiness' ? 'bg-yellow-400' :
                          emotion === 'sadness' ? 'bg-blue-400' :
                          emotion === 'surprise' ? 'bg-purple-400' : 'bg-gray-400'
                        }`}
                        style={{ width: `${value}%` }}
                      ></div>
                    </div>
                    <span className="text-xs ml-2 w-8 text-right">{value}%</span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}
      </div>
      
      <div className="flex justify-center space-x-4">
        {!cameraActive ? (
          <button 
            onClick={startCamera}
            className="bg-blue-600 hover:bg-blue-700 text-white py-2 px-6 rounded-full flex items-center"
          >
            <Camera size={18} className="mr-2" />
            Start Camera
          </button>
        ) : (
          <button 
            onClick={resetDemo}
            className="bg-gray-600 hover:bg-gray-700 text-white py-2 px-6 rounded-full flex items-center"
          >
            <RefreshCw size={18} className="mr-2" />
            Reset Demo
          </button>
        )}
      </div>
      
      <div className="mt-6 text-xs text-gray-500 text-center">
        <p>This is a demonstration only. No actual facial recognition is taking place.</p>
        <p>In a real implementation, this would use computer vision APIs to analyze facial features.</p>
      </div>
    </div>
  );
}