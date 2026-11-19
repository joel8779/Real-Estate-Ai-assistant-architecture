import React from 'react';

export const LoadingDots: React.FC = () => {
  return (
    <div className="flex items-center space-x-1.5 py-2 px-1">
      <div className="w-2 h-2 rounded-full bg-luxury-400 animate-bounce [animation-delay:-0.3s]" />
      <div className="w-2 h-2 rounded-full bg-luxury-400 animate-bounce [animation-delay:-0.15s]" />
      <div className="w-2 h-2 rounded-full bg-luxury-400 animate-bounce" />
    </div>
  );
};
