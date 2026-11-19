import '@testing-library/jest-dom';

// JSDOM does not implement scrollIntoView by default
window.HTMLElement.prototype.scrollIntoView = function() {};
